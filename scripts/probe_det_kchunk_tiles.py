"""Launch configs for deterministic K chunks: which tile and SPLIT_K is fastest, same bits.

    PYTHONPATH=src python scripts/probe_det_kchunk_tiles.py --shapes 4096x4096x4096,1x4096x4096

For each shape, deterministic mode: times the default config and a grid of tiles with SPLIT_K 1
and the largest power of two up to the chunk count, behind the idle gate (perf_gate.timed),
weights stored [K, N] (the prepacked layout); every config's output must equal the default's.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_gate  # noqa: E402
import probe_ieee_mm as P  # noqa: E402

TILES = [(128, 256, 16, 8, 3), (128, 128, 16, 4, 3), (256, 128, 16, 8, 3), (64, 128, 16, 4, 3),
         (128, 64, 16, 4, 3), (16, 64, 64, 2, 3), (32, 64, 32, 2, 4), (16, 128, 32, 2, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    perf_gate.wait_idle()
    torch = P.setup()
    from truestar import kernel as K
    from truestar.kernel import triton_backend as tb
    for M, N, Kd in P.parse_shapes(args.shapes):
        x, w, b = P.data(torch, M, N, Kd)
        wv = w.t().contiguous().t()
        kern = P.kernel_of(N, Kd)
        base = K.compile(kern, "cuda", tune=False, deterministic=True)
        ref = base(x=x, w=wv, b=b).clone()
        fns = {"default": lambda: base(x=x, w=wv, b=b)}
        nch = -(-Kd // tb.DET_K_CHUNK) if Kd > tb.DET_K_MIN else 1
        same = {}
        for t in TILES:
            for split in sorted({1, min(16, 1 << (nch.bit_length() - 1))}):
                c = tb._cfg(*t, split=split)
                if M <= 16 and t[0] > 32 or M > 64 and t[0] < 64:
                    continue
                ck = K.compile(kern, "cuda", deterministic=True, _config=c)
                name = f"{t[0]}x{t[1]}x{t[2]}w{t[3]}s{t[4]}/SK{split}"
                try:
                    same[name] = torch.equal(ck(x=x, w=wv, b=b), ref)
                except Exception as e:  # recorded
                    print(f"  {name}: {type(e).__name__} {str(e)[:100]}")
                    continue
                fns[name] = (lambda ck=ck: ck(x=x, w=wv, b=b))
        med, _, redone = perf_gate.timed(torch, fns, args.rounds)
        print(f"{M}x{N}x{Kd} default {base.launch_configs()[0][1]} {med['default']:.1f} us"
              f" (rounds redone {redone})", flush=True)
        for t, v in sorted(med.items(), key=lambda kv: kv[1])[:8]:
            print(f"   {t:28s} {v:9.1f}  {'' if t == 'default' or same.get(t) else 'BITS DIFFER'}",
                  flush=True)
        if not all(same.values()):
            print("   BITS DIFFER:", [t for t, s in same.items() if not s])


if __name__ == "__main__":
    main()
