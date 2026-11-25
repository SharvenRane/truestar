"""Unsplit decode attention launch configs: time and bits against the default config.

    PYTHONPATH=src python scripts/probe_decode_tiles.py

For a decode below the split threshold (one query row, or the query heads of one KV head folded
into rows) the default launch config is the prefill block (BM 64). This probe forces other BM,
num_warps and num_stages at the same BN (the key block, which decides the online softmax
rescaling points and so the bits) and checks every output against the default config's bits.
CUDA graph of 20 calls, median of rounds.
"""

import os
import statistics
import sys
import tempfile


def main():
    os.environ.setdefault("TRUESTAR_CACHE_DIR", tempfile.mkdtemp(prefix="ts_probe_dec_"))
    import torch
    import triton.testing
    from truestar import kernel as K
    from truestar.kernel import Graph

    D = 128

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

    cases = [(1, 8, 1, 512), (1, 8, 1, 1024), (1, 8, 4, 1024), (8, 8, 1, 1024), (1, 8, 4, 2000),
             (1, 32, 1, 1024), (2, 8, 16, 1536)]
    for det in (False, True):
        for B, H, T, N in cases:
            g = Graph()
            q_ = g.input("q", (B, H, K.sym("T"), D), "float16")
            k_ = g.input("k", (B, H, K.sym("N"), D), "float16")
            v_ = g.input("v", (B, H, K.sym("N"), D), "float16")
            (_, kern), = g.fuse({"o": g.softmax((q_ @ k_.T) * (D ** -0.5)) @ v_})
            gen = torch.Generator(device="cuda").manual_seed(N + T)
            q = torch.randn(B, H, T, D, device="cuda", generator=gen).half()
            k = torch.randn(B, H, N, D, device="cuda", generator=gen).half()
            v = torch.randn(B, H, N, D, device="cuda", generator=gen).half()
            base = K.compile(kern, "cuda", tune=False, deterministic=det)
            y0 = base(q=q, k=k, v=v).clone()
            d = base.launch_configs()[0][1]
            fns = {"default": lambda: base(q=q, k=k, v=v)}
            bad = []
            if d.get("KV_CHUNK", 0):
                print(f"det={det} B{B} H{H} T{T} N{N}: default splits ({d}), skipped", flush=True)
                continue
            for bm in (16, 32, 64):
                for w in (1, 2, 4, 8):
                    for st in (1, 2, 3, 4):
                        c = dict(d, BM=bm, num_warps=w, num_stages=st)
                        if c == d:
                            continue
                        try:
                            ck = K.compile(kern, "cuda", tune=False, deterministic=det, _config=c)
                            y = ck(q=q, k=k, v=v)
                            torch.cuda.synchronize()
                        except Exception as e:  # recorded, never hidden
                            print(f"   fail {c}: {type(e).__name__}: {str(e)[:100]}")
                            continue
                        if not torch.equal(y, y0):
                            bad.append((bm, w, st))
                        fns[(bm, w, st)] = (lambda ck=ck: ck(q=q, k=k, v=v))
            med = {}
            keys = list(fns)
            for i in range(0, len(keys), 16):
                graphs = {t: graph_of(fns[t]) for t in keys[i:i + 16]}
                times = {t: [] for t in graphs}
                for _ in range(3):
                    for t, gr in graphs.items():
                        times[t].append(triton.testing.do_bench(gr.replay, warmup=3, rep=15,
                                                                return_mode="median") / 20 * 1e3)
                med.update({t: statistics.median(x) for t, x in times.items()})
                del graphs
            ranked = sorted((v_, k_) for k_, v_ in med.items() if k_ != "default")
            print(f"det={det} B{B} H{H} T{T} N{N}: default BM{d['BM']} BN{d['BN']} w{d['num_warps']} "
                  f"s{d['num_stages']} {med['default']:.1f} us; configs {len(ranked)}, bit mismatches "
                  f"{len(bad)} {bad[:6]}", flush=True)
            for t, k_ in ranked[:4]:
                print(f"   BM, warps, stages {k_}: {t:.1f} us{'  BITS DIFFER' if k_ in bad else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
