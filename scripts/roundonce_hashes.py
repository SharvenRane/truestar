"""Hashes of round once output bits, to compare machines (docs/ROUND_ONCE.md, portability).

    PYTHONPATH=src python scripts/roundonce_hashes.py

Prints one JSON object: for each function, the SHA256 of the output bits over all float16 and all
bfloat16 inputs and over a fixed float32 sample of 2^22 bit patterns (seeded), computed by the torch
evaluator on the CPU, by the torch evaluator on CUDA, and by the Triton kernel path (float32 out)
on CUDA. Two machines agree bit for bit on a function exactly when their hashes are equal.
"""
import hashlib
import json
import os
import platform

os.environ.setdefault("TRUESTAR_VERIFY", "0")   # exp overflows float32 where float64 does not

import torch  # noqa: E402

from truestar import kernel as K  # noqa: E402
from truestar.kernel import ir, roundonce as R  # noqa: E402


def h(t):
    w = torch.int16 if t.element_size() == 2 else torch.int32
    return hashlib.sha256(t.contiguous().view(w).cpu().numpy().tobytes()).hexdigest()[:16]


def inputs():
    out = {f: torch.arange(65536, dtype=torch.int32).to(torch.int16).view(getattr(torch, f))
           for f in ("float16", "bfloat16")}
    g = torch.Generator().manual_seed(2026)
    bits = torch.randint(-(1 << 31), 1 << 31, (1 << 22,), generator=g, dtype=torch.int64)
    out["float32"] = bits.to(torch.int32).view(torch.float32)
    return out


def main():
    cuda = torch.cuda.is_available()
    res = {"machine": {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if cuda else None}}
    N = K.sym("N")
    i = K.axis("i", N)
    for name in R.FUNCTIONS:
        rec = {}
        for fmt, x in inputs().items():
            rec[f"{fmt}/torch_cpu"] = h(R.apply(name, x))
            if cuda:
                rec[f"{fmt}/torch_cuda"] = h(R.apply(name, x.cuda()))
                k = K.kernel(f"hash_{name}_{fmt}", [K.output("y", (i,), getattr(ir, name)(K.buffer("x", (N,), fmt)[i]), "float32")])
                y = K.compile(k, "cuda", math="round_once", tune=False)(x=x.cuda())
                y = torch.where(y == 0, torch.zeros_like(y), y)   # kernel stores -0 as +0
                ref = R.apply(name, x.cuda(), torch.float32)
                ref = torch.where(ref == 0, torch.zeros_like(ref), ref)
                rec[f"{fmt}/triton_f32"] = h(y)
                rec[f"{fmt}/torch_f32_plus_zero"] = h(ref)
        res[name] = rec
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
