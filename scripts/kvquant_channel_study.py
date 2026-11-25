"""Per channel against per token group key scales, at the same KV memory (truestar.kvquant).

    PYTHONPATH=src python scripts/kvquant_channel_study.py --json out.json

Keys per channel: one float32 per head, channel and block of D x g tokens; keys per group: one per
head and group of g tokens (g 32 by default). For D 128 both cost 4 / 32 bytes per token and head,
so the comparison is at equal memory; values always take one scale per group of g tokens.

Key distributions, each with q and v standard normal:

* ``normal``: standard normal keys.
* ``outlier_channels``: 4 of the D channels scaled by 20 in every token (the pattern reported for
  the keys of trained LLMs: a few channels carry large magnitudes).
* ``token_drift``: each token's keys scaled by a factor from 0.2 to 5 along the sequence (magnitudes
  change from token to token, which a block wide channel scale cannot follow).

Reported per case: the largest absolute error of the kernel's output against float64 attention over
the unquantized cache, for both layouts and both formats, and which wins. Random data, so the
outcome says which pattern each layout handles, not how a given trained model behaves.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--contexts", default="2048,8192")
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from truestar import kvquant as KQ

    H, D, g = 8, 128, a.group
    block = D * g                     # equal memory: D scales per block of D x g tokens
    rows = []
    for dist in ("normal", "outlier_channels", "token_drift"):
        for B in (int(x) for x in a.batches.split(",")):
            for N in (int(x) for x in a.contexts.split(",")):
                gen = torch.Generator(device="cuda").manual_seed(B * 100000 + N)
                q = torch.randn(B, H, 1, D, device="cuda", generator=gen).half()
                k = torch.randn(B, H, N, D, device="cuda", generator=gen)
                v = torch.randn(B, H, N, D, device="cuda", generator=gen).half()
                if dist == "outlier_channels":
                    k[..., :4] *= 20.0
                elif dist == "token_drift":
                    k *= torch.logspace(math_log10(0.2), math_log10(5.0), N, device="cuda")[:, None]
                k = k.half()
                ref = torch.softmax((q.double() @ k.double().transpose(-1, -2)) * D ** -0.5, -1) @ v.double()
                row = {"dist": dist, "B": B, "N": N, "group": g, "channel_block": block}
                for fmt in KQ.FORMATS:
                    vc = KQ.quantize(v, fmt, g)
                    for layout, kc in (("group", KQ.quantize(k, fmt, g)),
                                       ("channel", KQ.quantize(k, fmt, block, channel=True))):
                        out = KQ.decode_attention(q, kc, vc)
                        row[f"{fmt}_{layout}"] = (out.double() - ref).abs().max().item()
                    row[f"{fmt}_winner"] = ("channel" if row[f"{fmt}_channel"] < row[f"{fmt}_group"]
                                            else "group")
                rows.append(row)
                print(f"{dist:16} B{B} N{N}: " + "  ".join(
                    f"{f} group {row[f + '_group']:.2e} channel {row[f + '_channel']:.2e} -> {row[f + '_winner']}"
                    for f in KQ.FORMATS), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


def math_log10(x):
    import math
    return math.log10(x)


if __name__ == "__main__":
    main()
