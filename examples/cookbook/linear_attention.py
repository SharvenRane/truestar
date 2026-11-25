"""Linear attention models (Gated DeltaNet, Mamba 2 SSD, Kimi Delta Attention) on a verified kernel.

    python examples/cookbook/linear_attention.py        # CPU, about 10 s

`truestar.linattn.linear_attention` runs the chunked form of the recurrence (Triton on CUDA,
chunked PyTorch elsewhere) and checks its first call per signature against the float64 sequential
recurrence. The chunk size is a reduction order: any size is exact in real arithmetic, and this
script measures what it changes in float32 and how the error grows along the chain of chunks.
Recipe: docs/cookbook/linear_attention.md.
"""

import sys

import torch
import torch.nn.functional as F

from truestar.linattn import chain_error, chunk_invariance, last_call, linear_attention, recurrent

B, T, H, DK, DV = 2, 256, 4, 32, 32


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, DK, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, DK, generator=g), dim=-1)      # delta rule needs |k| = 1
    v = torch.randn(B, T, H, DV, generator=g)
    decay = -F.softplus(torch.randn(B, T, H, generator=g)) * 0.1          # log decay per head, <= 0
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g))               # write strength
    return q, k, v, decay, beta


def main():
    q, k, v, decay, beta = inputs()
    ok = True
    for name, args, rule in (("Gated DeltaNet (delta rule, scalar decay)", (q, k, v, decay, beta), "delta"),
                             ("Mamba 2 SSD (add rule, scalar decay)", (q, k, v, decay), "add")):
        o, state = linear_attention(*args, rule=rule, output_final_state=True)
        info = last_call()
        ref, ref_state = recurrent(*args, rule=rule)                    # float64, one step at a time
        rel = float((o.double() - ref).norm() / ref.norm())
        print(f"{name}\n  ran {info['impl']} with {info['config']}, checked against float64 on this call: "
              f"{info['checked']}\n  relative error vs the float64 recurrence: {rel:.1e}; final state "
              f"{tuple(state.shape)}")
        ok = ok and rel < 1e-5

    inv = chunk_invariance(q, k, v, decay, beta, rule="delta", chunks=(16, 32, 64))
    print("\nchunk size as a reduction order (delta rule, float32):")
    for c, e in inv["vs_float64"].items():
        print(f"  chunk {c:3}: worst element error vs float64 {e['o']:.1e}")
    for pair, d in inv["pairs"].items():
        print(f"  chunks {pair}: bitwise equal {d['bitwise_equal']}, {d['elements_differing']:.0%} of elements "
              f"differ, largest difference {d['max_abs']:.1e}")
    ch = chain_error(q, k, v, decay, beta, rule="delta", chunk=16)
    print(f"error along {ch['chunks']} chunks of 16: last quarter / first quarter = {ch['last_over_first']:.2f}"
          f" ({ch['code'] or 'no growth warning'})")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
