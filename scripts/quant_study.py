"""Quantize a small trained Llama style model per format and report both verdicts.

The model (truestar.rlkernels.llama: RMSNorm, rotary, grouped query attention, SwiGLU) is built in
code and trained for a few hundred steps on a synthetic order 2 Markov language, so it has a real
next token distribution to lose (no download). Then, per format:

* verdict (b), format loss: float64 format reference against the float64 float model (loss
  delta, perplexity ratio, KL, decision flips beyond near ties); no kernel involved;
* verdict (a), kernel: every quantized layer checked against the exact dequantized reference;
* the usual baseline: the quantized model against the float16 model.

Then the same with an injected kernel bug in one layer, to show (a) names it while (b) does not
move and the baseline cannot tell it from format loss.

    PYTHONPATH=src python scripts/quant_study.py --out quant_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from truestar.quant import model as QM
from truestar.rlkernels import llama as L

SPECS = ["int8", "int4:g128", "int4:g32", "fp8_e4m3:channel", "fp8_e4m3:tensor",
         "fp8_e5m2:channel", "nvfp4", "mxfp4"]


def markov_corpus(vocab: int, n: int, T: int, seed: int) -> torch.Tensor:
    """Markov chain: each previous token picks among 4 next tokens (seeded) with fixed
    probabilities 0.55, 0.25, 0.15, 0.05; one context in 4 instead copies the token two back.
    Entropy floor about 0.83 nats per token (1.11 on the Markov positions, 0 on the copies)."""
    g = torch.Generator().manual_seed(1234)
    cand = torch.randint(0, vocab, (vocab, 4), generator=g)
    probs = torch.tensor([0.55, 0.25, 0.15, 0.05])
    g2 = torch.Generator().manual_seed(seed)
    out = torch.empty(n, T, dtype=torch.long)
    out[:, :2] = torch.randint(0, vocab, (n, 2), generator=g2)
    for t in range(2, T):
        prev = out[:, t - 1]
        pick = torch.multinomial(probs.expand(n, 4), 1, generator=g2).squeeze(1)
        nxt = cand[prev, pick]
        out[:, t] = torch.where(prev % 4 == 0, out[:, t - 2], nxt)
    return out


def train(cfg: L.Config, steps: int, seed: int = 0) -> L.Llama:
    torch.manual_seed(seed)
    m = L.Llama(cfg).cuda().float().train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    data = markov_corpus(cfg.vocab, 8192, 129, seed=1)
    t0 = time.time()
    for s in range(steps):
        i = torch.randint(0, data.shape[0], (32,))
        ids = data[i].cuda()
        logits = m(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"step {s} loss {loss.item():.3f} ({time.time() - t0:.0f} s)", file=sys.stderr)
    return m.eval()


def summarize(r: dict) -> str:
    fv, kv, bv = r["format_verdict"], r["kernel_verdict"], r["baseline_vs_fp16"]
    f = fv["flips"]
    k = kv["vs_format_reference_flips"]
    b = bv["flips"]
    s = (f"{r['format']:18s} bytes {r['weight_bytes'] / 1e6:6.2f} MB | (b) dCE {fv['delta_ce']:+.4f} "
         f"ppl x{fv['ppl_ratio']:.4f} KL {fv['kl']:.4f} flips {f['flips']}/{f['total']} "
         f"({f['near_ties']} near ties) | (a) {'ok' if kv['ok'] else 'FAIL'} worst {kv['worst_ratio']:.3g}"
         f" kernel vs format flips {k['flips']} ({k['near_ties']} ties) | baseline vs fp16 "
         f"dCE {bv['delta_ce_vs_fp16']:+.4f} rel {bv['rel']:.3g} flips {b['flips']}")
    for c in kv["failing"]:
        s += f"\n    {c['code']} {c['layer']}: ratio {c['ratio']:.3g}; " + "; ".join(c["diagnosis"])
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()
    cfg = L.Config(vocab=512, dim=512, layers=4, heads=8, kv_heads=4, hidden=1536, max_len=256)
    m = train(cfg, args.steps)
    dt = getattr(torch, args.dtype)
    m16 = m.to(dt)
    ids = markov_corpus(cfg.vocab, 16, 129, seed=99).cuda()      # held out: 16 x 128 positions
    t0 = time.time()
    res = QM.compare(m16, ids, SPECS)
    print(f"float64 CE {res['ce_float64']:.4f} (entropy floor about 0.83), fp16 CE {res['ce_fp16']:.4f}, "
          f"{res['positions']} positions, {time.time() - t0:.1f} s")
    for r in res["formats"]:
        print(summarize(r))
    injected = []
    for spec, layer, bug in [("int4:g128", "blocks.2.mlp.down_proj", "last_group_scale"),
                             ("int4:g128", "blocks.1.attn.o_proj", "group_scale_shift"),
                             ("nvfp4", "blocks.0.mlp.up_proj", "block_scale_shift"),
                             ("fp8_e4m3:channel", "blocks.3.attn.q_proj", "tensor_scale_for_channel")]:
        r = QM.compare(m16, ids, [spec], inject={layer: bug}, layer_inputs=False)["formats"][0]
        print(f"INJECTED {bug} in {layer}:")
        print(summarize(r))
        injected.append({"spec": spec, "layer": layer, "bug": bug, "result": r})
    if args.out:
        def clean(d):
            if isinstance(d, dict):
                return {k: clean(v) for k, v in d.items() if not k.startswith("_")}
            if isinstance(d, list):
                return [clean(v) for v in d]
            return d
        with open(args.out, "w") as f:
            json.dump(clean({"config": cfg.__dict__, "steps": args.steps, "dtype": args.dtype,
                             "result": res, "injected": injected}), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
