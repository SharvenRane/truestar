"""Does a combined device and dtype copy from or to MPS give the same values as two steps?

Found while building the Metal backend (docs/METAL.md): on torch 2.14.1 on GitHub's macOS 15
runner, ``t.to("cpu", torch.float64)`` of MPS tensors written by a Metal kernel returned values no
float32 can hold (2.8e-288). This probe measures it on plain torch tensors and on tensors written
by a kernel compiled with torch.mps.compile_shader, for a few shapes, in both directions.

    python scripts/mps_copy_probe.py
"""

import torch

SRC = """
kernel void copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) { y[i] = x[i]; }
"""

SHAPES = [(), (7,), (13, 21), (2, 6, 8), (37, 45), (9, 300), (11, 96), (1, 411), (2, 2, 24, 16)]


def bad(a, b):
    return int((a != b).sum()) if a.numel() else 0


def main():
    print("torch", torch.__version__)
    lib = torch.mps.compile_shader(SRC)
    g = torch.Generator().manual_seed(0)
    total = 0
    for shape in SHAPES:
        x = torch.randn(shape, generator=g)
        xm = x.to("mps")
        ym = torch.empty_like(xm)
        n = max(1, x.numel())
        lib.copy(xm, ym, threads=[n, 1, 1], group_size=[min(n, 64), 1, 1])
        for name, t in (("torch op", xm * 1.0), ("kernel output", ym)):
            one = t.to("cpu", torch.float64)
            two = t.cpu().to(torch.float64)
            k = bad(one, two)
            total += k
            if shape == (7,):
                print("  combined:", one[:3].tolist(), " two steps:", two[:3].tolist())
            print(f"MPS -> CPU float64 {name:14s} {str(shape):14s} combined differs at {k} of {x.numel()}")
        for dt in (torch.float16, torch.bfloat16):
            one = x.to("mps", dt).cpu()
            two = x.to(dt).to("mps").cpu()
            k = bad(one, two)
            total += k
            print(f"CPU -> MPS {str(dt):14s} {str(shape):14s} combined differs at {k} of {x.numel()}")
    print("differing elements in all:", total)


if __name__ == "__main__":
    main()
