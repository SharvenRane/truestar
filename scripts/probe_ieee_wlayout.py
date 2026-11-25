"""Does Truestar's own strict float32 contraction get faster, with the same bits, when the weight
is stored [K, N] (passed as the view w_kn.t()) or the input is stored [K, M]?

    PYTHONPATH=src python scripts/probe_ieee_wlayout.py
"""
import sys
sys.path.insert(0, "scripts")
import probe_ieee_mm as P  # noqa: E402


def main():
    torch = P.setup()
    import torch.nn.functional as F
    from truestar import kernel as K
    print(f"gpu {torch.cuda.get_device_name(0)}")
    for M, N, Kd in P.SHAPES:
        x, w, b = P.data(torch, M, N, Kd)
        ref = torch.relu(x.double() @ w.double().T + b.double())
        wv = w.t().contiguous().t()      # same values, stored [K, N]
        xv = x.t().contiguous().t()      # same values, stored [K, M]
        kern = P.kernel_of(N, Kd)
        fns, outs = {}, {}
        for det in (True, False):
            ck = K.compile(kern, "cuda", tune=False, deterministic=det)
            tag = "det" if det else "def"
            for name, xx, ww in (("nk", x, w), ("kn", x, wv), ("kn_xt", xv, wv)):
                f = (lambda ck=ck, xx=xx, ww=ww: ck(x=xx, w=ww, b=b))
                fns[f"{tag}_{name}"] = f
                outs[f"{tag}_{name}"] = f().clone()
        fns["F.linear"] = lambda: torch.relu(F.linear(x, w, b))
        med, _ = P.timed(torch, fns, 5)
        same = {t: torch.equal(outs[t], outs[t[:3] + "_nk"]) for t in outs}
        err = max(P.rel_err(o, ref) for o in outs.values())
        cub = med["F.linear"]
        print(f"{M}x{N}x{Kd} F.linear {cub:.1f} | " + "  ".join(
            f"{t} {med[t]:.1f} ({cub / med[t]:.2f}x{'' if same[t] else ' BITS DIFFER'})" for t in outs)
            + f" | max err {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
