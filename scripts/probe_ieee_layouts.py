"""Strict float32 operand layout probe: does the weight layout or a hand written FMA loop beat the
current tl.dot (ieee) kernel, with the same bits?

    PYTHONPATH=src python scripts/probe_ieee_layouts.py

Variants, all one FMA chain per output over K in order (no split K):
  nk   : W stored [N, K] (as the importer gives it), B tile read transposed, tl.dot ieee
  kn   : W stored [K, N] (pre transposed once), tl.dot ieee
  xt   : x stored [K, M] and W [K, N], tl.dot ieee
Times are CUDA graph medians (triton.testing.do_bench_cudagraph), cuBLAS is torch.mm with TF32 off.
"""
import itertools
import torch
import triton
import triton.language as tl


@triton.jit
def mm_kernel(x, w, y, M, N, K, sxm, sxk, swk, swn,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nb = tl.cdiv(N, BN)
    pm = pid // nb
    pn = pid % nb
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x + rm[:, None] * sxm + (k0 + rk)[None, :] * sxk,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(w + (k0 + rk)[:, None] * swk + rn[None, :] * swn,
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(y + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(x, w, layout, cfg):
    BM, BN, BK, nw, ns = cfg
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, device="cuda")
    if layout == "nk":
        xx, ww = x, w
        args = (xx.stride(0), xx.stride(1), ww.stride(1), ww.stride(0))
    elif layout == "kn":
        xx, ww = x, w.t().contiguous()
        args = (xx.stride(0), xx.stride(1), ww.stride(0), ww.stride(1))
    else:  # xt
        xx, ww = x.t().contiguous(), w.t().contiguous()
        args = (xx.stride(1), xx.stride(0), ww.stride(0), ww.stride(1))
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def f():
        mm_kernel[grid](xx, ww, y, M, N, K, *args, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    f()
    return y, f


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    print("gpu", torch.cuda.get_device_name(), "triton", triton.__version__)
    shapes = [(1024, 1024, 1024), (2048, 4096, 1024), (4096, 4096, 4096), (64, 2048, 2048),
              (8192, 768, 256), (8192, 256, 1024)]
    cfgs = [(128, 128, 16, 4, 3), (128, 256, 16, 8, 3), (256, 128, 16, 8, 3), (64, 128, 16, 4, 3),
            (128, 64, 16, 4, 3), (128, 128, 32, 8, 2), (64, 64, 32, 4, 3), (64, 16, 32, 1, 3)]
    for (M, N, K) in shapes:
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(M, K, device="cuda", generator=g)
        w = torch.randn(N, K, device="cuda", generator=g) / K ** 0.5
        ref64 = (x.double() @ w.double().t())
        cub = triton.testing.do_bench_cudagraph(lambda: torch.mm(x, w.t()))
        base, _ = run(x, w, "nk", cfgs[0])
        err = ((base.double() - ref64).abs().max() / ref64.abs().max()).item()
        print(f"\n{M}x{N}x{K} cuBLAS {cub*1e3:.1f} us  (ts nk 128x128x16 rel err {err:.1e})")
        for cfg, layout in itertools.product(cfgs, ["nk", "kn", "xt"]):
            try:
                y, f = run(x, w, layout, cfg)
                t = triton.testing.do_bench_cudagraph(f)
            except Exception as e:  # out of shared memory etc.
                print(f"  {layout} {cfg}: fail {type(e).__name__}")
                continue
            same = torch.equal(y, base)
            print(f"  {layout} {cfg}: {t*1e3:8.1f} us  {cub/t:5.2f}x cuBLAS  bits {'same' if same else 'DIFF'}")


if __name__ == "__main__":
    main()
