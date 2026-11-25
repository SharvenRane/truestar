"""Quantize safely: measure what int8 and fp8 cost against float64, and verify the kernel that runs them.

    python examples/cookbook/quantize_safely.py        # CPU, about 5 s

Two separate questions, two verdicts (docs/QUANT.md):

* (b) the format: how far do the quantized weights (and fp8 activations) move each layer and the
  predictions, computed in float64 so no kernel is involved. A prediction that changes where the
  float64 top two were within the float32 model's own rounding is a near tie, reported apart;
  the other changes are the format's loss, yours to accept or not.
* (a) the kernel: does the matmul that runs the quantized layer compute what the format says,
  within a bound derived from float64. Here the kernels are short CPU functions written in this
  file; one correct, one with a common bug (the per channel scale stored in bfloat16).
Recipe: docs/cookbook/quantize_safely.md.
"""

import sys

import torch

from truestar import nn
from truestar.quant import formats as F
from truestar.quant.model import decisions
from truestar.quant.verify import check_matmul, layer_format_loss, quantize_act_ref

D, H, C, N = 128, 256, 10, 512
net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, C))
LINEARS = ["0", "2", "4"]


def cast(tree, dtype):
    return {k: cast(v, dtype) for k, v in tree.items()} if isinstance(tree, dict) else tree.to(dtype)


def forward_with(p, x, fmt=None):
    """The model in float64 with every linear layer's weight replaced by its quantized value, and
    with fp8 its input quantized per row as well. Returns the logits and each linear's input."""
    ins, h = {}, x
    for i, layer in enumerate(net.layers):
        k = str(i)
        if k in LINEARS:
            ins[k] = h
            w, b = p[k]["w"], p[k]["b"]
            if fmt is not None:
                w = F.quantize_weight(w.float(), fmt.spec).dequantize()
                if fmt.activations:
                    h = quantize_act_ref(h, fmt)
            h = h @ w.T + b
        else:
            h = layer(h)
    return h, ins


def int8_kernel(x, qw, scale_dtype=torch.float32):
    """y = (x @ codes^T) * scale, accumulated in float32. scale_dtype=bfloat16 is the bug."""
    codes = qw.data.to(torch.float32)
    return (x.float() @ codes.T) * qw.scale.to(scale_dtype).float()


def fp8_kernel(x, qw):
    """W8A8 fp8: quantize each activation row (amax / 448, e4m3), multiply, rescale."""
    amax = x.float().abs().amax(-1).clamp(min=2.0 ** -126)
    s = (amax / 448.0).float()
    xa = (x.float() / s[:, None]).to(torch.float8_e4m3fn)
    act = F.QuantAct(qw.fmt, xa, s)
    y = (xa.float() @ qw.data.float().T) * s[:, None] * qw.scale.float()
    return y, act


def main():
    p64 = cast(net.init(seed=0), torch.float64)
    x = torch.randn(N, D, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    ref, ins = forward_with(p64, x)
    # A near tie is a row whose float64 top two are within 4 times the compute noise of the plain
    # float32 model on that row (the rule truestar.quant.model.compare uses).
    out32, _ = forward_with(cast(p64, torch.float32), x.float())
    tie = 4 * (out32.double() - ref).abs().amax(-1)
    ok = True

    print("(b) the format alone, in float64")
    for spec in ("int8", "fp8_e4m3:channel"):
        fmt = F.parse(spec)
        out, _ = forward_with(p64, x, fmt)
        per_layer = []
        for k in LINEARS:
            qw = F.quantize_weight(p64[k]["w"].float(), spec)
            per_layer.append(f"{k}: {layer_format_loss(ins[k], p64[k]['w'], qw):.1e}")
        err = float((out - ref).norm() / ref.norm())
        d = decisions(ref, out, tie)
        print(f"  {spec:18} layer error {', '.join(per_layer)}; output {err:.1e}; "
              f"top 1 changed {d.flips} of {d.total} ({d.near_ties} near ties, {d.real} real)")

    print("\n(a) the kernel, against the float64 reference of the same codes and scales")
    xl, wl = ins["2"].float(), p64["2"]["w"].float()
    q8 = F.quantize_weight(wl, "int8")
    for name, y in (("int8 kernel", int8_kernel(xl, q8)),
                    ("int8 kernel, scale in bfloat16", int8_kernel(xl, q8, torch.bfloat16))):
        chk = check_matmul(xl, q8, y, layer="2")
        print(f"  {name:32} ok {chk.ok!s:5} error / bound {chk.ratio:8.2f}  {chk.code or ''}")
        ok = ok and chk.ok == (name == "int8 kernel")
    q8f = F.quantize_weight(wl, "fp8_e4m3:channel")
    y, act = fp8_kernel(xl, q8f)
    chk = check_matmul(xl, q8f, y, act, layer="2")
    print(f"  {'fp8 e4m3 W8A8 kernel':32} ok {chk.ok!s:5} error / bound {chk.ratio:8.2f}  "
          f"activation codes off the format: {chk.quantizer_mismatches}")
    ok = ok and chk.ok
    print("\nOK: the correct kernels pass, the bfloat16 scale is caught" if ok else "UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
