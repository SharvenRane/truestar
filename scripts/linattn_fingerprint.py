"""Bitwise fingerprint of the deterministic linear attention kernels on this GPU.

Run on two GPUs of one architecture with different SM counts (RTX 5070 Ti, 70 SMs; RTX 5060 Laptop
GPU, 26 SMs) and compare the printed JSON: in deterministic mode every digest must be equal.

    PYTHONPATH=src python scripts/linattn_fingerprint.py > fp.json

Inputs are drawn on the CPU from fixed seeds, so both machines see the same bits.
"""

import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from truestar import linattn as LA  # noqa: E402

CASES = [  # B, T, H, dk, dv, rule, decay
    (1, 4096, 8, 128, 64, "add", "scalar"),
    (2, 1000, 4, 128, 128, "delta", "scalar"),
    (4, 777, 3, 64, 64, "delta", "none"),
    (3, 300, 2, 48, 40, "add", "none"),
]


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    out = {"device": torch.cuda.get_device_name(0),
           "sms": torch.cuda.get_device_properties(0).multi_processor_count, "cases": {}}
    for i, (B, T, H, dk, dv, rule, decay) in enumerate(CASES):
        gen = torch.Generator().manual_seed(100 + i)
        q = torch.randn(B, T, H, dk, generator=gen) * dk ** -0.5
        k = torch.nn.functional.normalize(torch.randn(B, T, H, dk, generator=gen), dim=-1)
        v = torch.randn(B, T, H, dv, generator=gen)
        g = -torch.rand(B, T, H, generator=gen) * 0.3 if decay == "scalar" else None
        beta = torch.sigmoid(torch.randn(B, T, H, generator=gen)) if rule == "delta" else None
        q, k, v, g, beta = (None if x is None else x.cuda() for x in (q, k, v, g, beta))
        o, S = LA.linear_attention(q, k, v, g, beta, rule=rule, deterministic=True, output_final_state=True)
        last = LA.last_call()
        out["cases"][f"{rule}/{decay}/{B}x{T}x{H}x{dk}x{dv}"] = {
            "o": digest(o), "state": digest(S), "config": last["config"], "error_vs_float64": last["errors"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
