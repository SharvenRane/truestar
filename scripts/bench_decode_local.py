"""scripts/bench_decode_flashinfer.py behind the idle gate (perf_gate.py), for a local GPU without
FlashInfer, plus a determinism and float64 pass over the quantized decode.

    PYTHONPATH=src python scripts/bench_decode_local.py [bench_decode_flashinfer.py arguments]

Before timing, every quantized case (int8 and fp8, both modes) runs twice and must give equal
bits; decode_attention's first call check (float64 attention over the dequantized cache) runs on
each and raises on a mismatch."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_gate  # noqa: E402


def det_pass(args):
    import torch
    from truestar import kvquant as KQ
    heads = [tuple(int(v) for v in hp.split(",")) for hp in args.get("--heads", "32,8;8,8").split(";")]
    D = 128
    n_ok = 0
    for Hq, Hkv in heads:
        for B in (int(x) for x in args.get("--batches", "1,8").split(",")):
            for N in (int(x) for x in args.get("--contexts", "1024,4096,16384").split(",")):
                g = torch.Generator(device="cuda").manual_seed(1)
                q = torch.randn(B, Hq, 1, D, device="cuda", generator=g).half()
                k = torch.randn(B, Hkv, N, D, device="cuda", generator=g).half()
                v = torch.randn(B, Hkv, N, D, device="cuda", generator=g).half()
                for fmt in ("int8", "fp8"):
                    kc, vc = KQ.quantize(k, fmt, group=32), KQ.quantize(v, fmt, group=32)
                    for det in (False, True):
                        a = KQ.decode_attention(q, kc, vc, deterministic=det)
                        b = KQ.decode_attention(q, kc, vc, deterministic=det)
                        ref = KQ.reference(q, kc, vc)
                        err = ((a.double() - ref).abs().max() / ref.abs().max()).item()
                        assert torch.equal(a, b), (Hq, Hkv, B, N, fmt, det)
                        n_ok += 1
                        print(f"  det pass Hq{Hq}/{Hkv} B{B} N{N} {fmt} det={det}: runs equal, "
                              f"err vs float64 over the dequantized cache {err:.2e}", flush=True)
    print(f"determinism pass: {n_ok} cases, two runs equal in all", flush=True)


def main():
    argv = sys.argv[1:]
    args = dict(zip(argv[::2], argv[1::2]))
    perf_gate.wait_idle()
    det_pass(args)
    import bench_decode_flashinfer
    sys.argv = [sys.argv[0]] + argv
    bench_decode_flashinfer.main()


if __name__ == "__main__":
    main()
