"""Measured numbers for the low precision formats work (docs/QUANT.md, ledger row P50).

1. MX scale rules (floor, floor_adj, ceil) on every MX element format: relative MSE against
   float64 and clipped elements, on the linear weights of a real model and on heavy tailed
   synthetic data.
2. The format loss (verdict b) of W4A4 NVFP4 and MXFP4 on the model.
3. Keyed stochastic rounding: bias over many draws, bitwise equality CPU / CUDA torch / Triton,
   speed of the Triton kernel against round to nearest casts, and a bfloat16 accumulation.
4. MXFP8 matmul (tl.dot_scaled) verdict and speed against a float16 matmul.

    PYTHONPATH=src python scripts/lowp_study.py --out results/lowp/study.json [--skip-model]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from truestar.quant import formats as F
from truestar.quant import mx
from truestar.quant import sround as S

MODEL = "HuggingFaceTB/SmolLM2-360M"


def _linears(model):
    return [(n, m.weight.detach()) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


def _rel_mse(d, w):
    w = w.double()
    return float(((d - w) ** 2).sum() / (w ** 2).sum())


def mx_rules(weights):
    out = {}
    g = torch.Generator().manual_seed(0)
    synth = torch.randn(512, 1024, generator=g) * torch.exp(torch.randn(512, 1024, generator=g))
    for fmt in mx.MX_FORMATS:
        row = {}
        for rule in mx.SCALE_RULES:
            err = n = clip = 0.0
            tot = 0
            for _, w in weights:
                q = mx.quantize(w, fmt, rule)
                err += float(((q.dequantize() - w.double()) ** 2).sum())
                n += float((w.double() ** 2).sum())
                clip += q.clipped(w)
                tot += w.numel()
            qs = mx.quantize(synth, fmt, rule)
            row[rule] = {"model_rel_mse": err / n, "model_clipped": int(clip), "model_elements": tot,
                         "synthetic_rel_mse": _rel_mse(qs.dequantize(), synth),
                         "synthetic_clipped": qs.clipped(synth)}
        out[fmt] = row
    return out


class _RefLinear(torch.nn.Module):
    """float64 format reference of one linear: dequantized weight, activations quantized by the
    format's rule (verify.quantize_act_ref)."""

    def __init__(self, lin, spec):
        super().__init__()
        from truestar.quant import verify as VV
        self.qw = F.quantize_weight(lin.weight.detach(), spec)
        self.w = self.qw.dequantize()
        self.b = None if lin.bias is None else lin.bias.detach().double()
        self.fmt = self.qw.fmt
        self.VV = VV

    def forward(self, x):
        xq = self.VV.quantize_act_ref(x, self.fmt)
        y = xq @ self.w.T
        return y if self.b is None else y + self.b


@torch.no_grad()
def model_format_loss(model, tok, ntok=256, nseq=4):
    """Verdict (b), the format loss, of W4A4 on the whole model in float64 on the CPU: cross
    entropy change, KL and top 1 flips against the float64 float model on WikiText 2 test."""
    import copy
    files = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/"
                                         "snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
    import pandas as pd
    text = chr(10).join(t for t in pd.read_parquet(files[0])["text"].tolist() if t.strip())
    ids = tok(text[:200000], return_tensors="pt").input_ids[0]
    ids = torch.stack([ids[i * ntok:(i + 1) * ntok] for i in range(nseq)])
    f64 = copy.deepcopy(model).double().eval()
    tgt = ids[:, 1:]

    def run(m):
        return m(input_ids=ids).logits[:, :-1].double()

    def ce(L):
        return float(torch.nn.functional.cross_entropy(L.reshape(-1, L.shape[-1]), tgt.reshape(-1)))

    L64 = run(f64)
    p64 = torch.log_softmax(L64, -1)
    out = {"model": MODEL, "positions": int(tgt.numel()), "ce_float64": ce(L64), "formats": {}}
    for name, spec in (("nvfp4", "nvfp4"), ("mxfp4", "mxfp4")):
        r = copy.deepcopy(f64)
        for n, w in _linears(model):
            parent, _, leaf = n.rpartition(".")
            setattr(r.get_submodule(parent) if parent else r, leaf,
                    _RefLinear(model.get_submodule(n), spec))
        L = run(r)
        pf = torch.log_softmax(L, -1)
        out["formats"][name] = {
            "ce": ce(L), "ce_delta": ce(L) - out["ce_float64"],
            "kl": float((p64.exp() * (p64 - pf)).sum(-1).mean()),
            "top1_flips": int((L.argmax(-1) != L64.argmax(-1)).sum())}
        del r
    return out


def _time(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


def sround_numbers(gpu: bool):
    out = {"bias": {}, "bitwise": {}}
    g = torch.Generator().manual_seed(1)
    for t, hi in (("e2m1", 5.9), ("e4m3", 440.0), ("e5m2", 5e4), ("bf16", 1e4), ("fp16", 1e4)):
        x = (torch.rand(4000, generator=g) * 2 - 1) * hi
        bt = S.bias_test(x, t, draws=400)
        out["bias"][t] = {"mean_err_quanta": bt.mean_err, "z": bt.z, "var_ratio": bt.var_ratio,
                          "worst_elem_z": bt.max_abs_elem_z, "ok": bt.ok, "draws": bt.draws,
                          "elements": bt.elements}
    x = torch.randn(1 << 20, generator=g) * 4
    for t in ("e2m1", "e4m3", "bf16"):
        ref = S.round_torch(x, t, 9, "b", 0)
        row = {"python_first_4096": bool(torch.equal(S.round_python(x[:4096], t, 9, "b", 0), ref[:4096]))}
        if gpu:
            row["torch_cuda"] = bool(torch.equal(S.round_torch(x.cuda(), t, 9, "b", 0).cpu(), ref))
            row["triton"] = bool(torch.equal(S.round_triton(x.cuda(), t, 9, "b", 0).cpu(), ref))
        out["bitwise"][t] = row
    if gpu:
        xg = torch.randn(1 << 24, device="cuda")
        out["speed_us_16M"] = {
            "triton_sr_bf16_codes": _time(lambda: S.round_triton(xg, "bf16", 1, "s", 0)),
            "torch_ops_sr_bf16_codes": _time(lambda: S.round_torch(xg, "bf16", 1, "s", 0), 5),
            "rtn_cast_bf16": _time(lambda: xg.to(torch.bfloat16)),
            "triton_sr_e4m3_codes": _time(lambda: S.round_triton(xg, "e4m3", 1, "s", 0)),
            "rtn_cast_e4m3": _time(lambda: xg.to(torch.float8_e4m3fn)),
        }
    acc_rn = torch.ones(1024, dtype=torch.bfloat16)
    acc_sr = torch.ones(1024, dtype=torch.bfloat16)
    for s in range(10000):
        acc_rn = (acc_rn.float() + 1e-3).bfloat16()
        acc_sr = S.to_dtype(acc_sr.float() + 1e-3, torch.bfloat16, 5, "acc", s, backend="torch")
    out["bf16_accumulate_1e-3_x10000_from_1"] = {
        "float64": 1.0 + 10000 * 1e-3, "round_to_nearest": float(acc_rn.float().mean()),
        "keyed_sr_mean": float(acc_sr.float().mean()),
        "keyed_sr_std": float(acc_sr.float().std())}
    return out


def mxfp8_numbers():
    from truestar.quant import mx_kernels as K
    out = {}
    g = torch.Generator().manual_seed(2)
    M = N = K_ = 4096
    x = torch.randn(M, K_, generator=g)
    w = torch.randn(N, K_, generator=g) * 0.02
    for f in ("mxfp8_e4m3", "mxfp8_e5m2"):
        qa, qb = mx.quantize(x, f), mx.quantize(w, f)
        qa = mx.MXTensor(qa.fmt, qa.codes.cuda(), qa.scale_codes.cuda())
        qb = mx.MXTensor(qb.fmt, qb.codes.cuda(), qb.scale_codes.cuda())
        y = K.matmul(qa, qb)
        sub = mx.MXTensor(qa.fmt, qa.codes[:256], qa.scale_codes[:256])
        ratio = K.verdict(sub, qb, y[:256])
        fmt_err = float((qa.dequantize()[:256] @ qb.dequantize().T - x[:256].double().cuda() @ w.double().cuda().T).norm()
                        / (x[:256].double().cuda() @ w.double().cuda().T).norm())
        out[f] = {"verdict_ratio_first_256_rows": ratio, "format_rel_err_first_256_rows": fmt_err,
                  "us": _time(lambda: K.matmul(qa, qb), 10)}
    a16, b16 = x.cuda().half(), w.cuda().half()
    out["fp16_matmul_us"] = _time(lambda: a16 @ b16.T, 10)
    out["shape"] = [M, N, K_]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lowp/study.json")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()
    gpu = torch.cuda.is_available() and not a.skip_gpu
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name() if gpu else None}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    weights = _linears(model)
    res["model"] = MODEL
    t = time.time()
    # MX rules on the linears of 4 of the 32 blocks (0, 10, 20, 31) to keep the CPU time short
    sub = [(n, w) for n, w in weights if any(f"layers.{i}." in n for i in (0, 10, 20, 31))]
    res["mx_rules_layers"] = sorted({n.split(".")[2] for n, _ in sub})
    res["mx_rules"] = mx_rules(sub)
    print("cpu parts", round(time.time() - t, 1), "s", flush=True)
    res["sround"] = sround_numbers(gpu)
    if gpu:
        res["mxfp8_matmul"] = mxfp8_numbers()
    if not a.skip_model:
        res["model_format_loss"] = model_format_loss(model, tok)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
