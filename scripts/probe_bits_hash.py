"""Hash of strict float32 contraction outputs, to compare two source trees bit for bit.

    PYTHONPATH=<src A> python scripts/probe_bits_hash.py > a.txt
    PYTHONPATH=<src B> python scripts/probe_bits_hash.py > b.txt     # then diff a.txt b.txt

Kernel layer relu(x @ W.T + b), weight [N, K], tuning off, deterministic and default mode, plus
quantized decode attention (int8, fp8) in both modes. Same seeded data in both runs.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_ieee_mm as P  # noqa: E402


def h(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    torch = P.setup()
    from truestar import kernel as K
    for M, N, Kd in P.SHAPES + [(33, 200, 1500), (300, 1024, 4096)]:
        x, w, b = P.data(torch, M, N, Kd)
        kern = P.kernel_of(N, Kd)
        for det in (True, False):
            ck = K.compile(kern, "cuda", tune=False, deterministic=det)
            print(f"mm {M}x{N}x{Kd} det={det} {h(ck(x=x, w=w, b=b))}", flush=True)
    from truestar import kvquant as KQ
    for Hq, Hk, B, N in ((32, 8, 1, 1024), (8, 8, 8, 4096), (8, 8, 1, 300), (4, 4, 1, 1024)):
        g = torch.Generator(device="cuda").manual_seed(2)
        q = torch.randn(B, Hq, 1, 128, device="cuda", generator=g).half()
        k = torch.randn(B, Hk, N, 128, device="cuda", generator=g).half()
        v = torch.randn(B, Hk, N, 128, device="cuda", generator=g).half()
        for fmt in ("int8", "fp8"):
            kc, vc = KQ.quantize(k, fmt), KQ.quantize(v, fmt)
            for det in (True, False):
                print(f"kvq {Hq}/{Hk} B{B} N{N} {fmt} det={det} {h(KQ.decode_attention(q, kc, vc, deterministic=det))}",
                      flush=True)


if __name__ == "__main__":
    main()
