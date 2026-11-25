"""KV cache below 8 bits on a real model's keys and values (truestar.kvquant int4, nvfp4, rotated
or not), next to int8 and fp8.

    PYTHONPATH=src python scripts/kvquant_low_study.py --json results/rotation/kv_study.json

Model: SmolLM2-360M (32 layers, 5 KV heads of dim 64, 15 query heads) from the local Hugging Face
cache; text: WikiText-2 raw test split.

1. Per layer decode check: the keys (after RoPE) and values of a 2048 token prefix and the query
   of the last token are captured; the cache is quantized and the decode kernel runs over it.
   Reported: the largest absolute error of the output against float64 attention over the float16
   cache, the proven ``error_bound`` on the same data, and their ratio, over the 32 layers.
2. Perplexity with the whole cache in each format (keys after RoPE and values replaced by what the
   cache would hold, every layer; float32 model on the CPU, quantization on the GPU with the same
   quantizer the kernel reads).
3. Bytes per token for this model and for 8 KV heads of dim 128.
4. ``--bench``: decode attention speed on the GPU (synthetic, 32 query and 8 KV heads of dim 128),
   briefly: float16 SDPA against every format.
"""

import argparse
import json
import os

FMTS = [("float16", False), ("int8", False), ("fp8", False), ("int4", False), ("int4", True),
        ("nvfp4", False), ("nvfp4", True)]


def label(fmt, rot):
    return fmt + (" rotated" if rot else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--ppl", default="2x1024")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    import torch
    from transformers.models.llama import modeling_llama as ML
    from truestar import kvquant as KQ
    from rotation_study import load, text_ids

    torch.set_num_threads(a.threads)
    tok, model = load(a.model)
    cfg = model.config
    Hk, hd, Hq = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_attention_heads
    out = {"model": a.model, "group": a.group}

    def cache_values(x, fmt, rot):
        """What a cache in fmt holds for x [B, H, N, D] (float32 on the CPU), in x's basis."""
        x16 = x.half().cuda()
        if fmt == "float16":
            return x16.float().cpu()
        return KQ.quantize(x16, fmt, a.group, rot=rot).original().float().cpu()

    # 1. per layer decode check on captured K, V, q
    ids = text_ids(tok, "test", 1, a.prefix + 1)
    caps = {}
    orig = ML.apply_rotary_pos_emb

    def grab(q, k, cos, sin, *rest, **kw):
        qe, ke = orig(q, k, cos, sin, *rest, **kw)
        caps.setdefault("qk", []).append((qe[:, :, -1:].detach(), ke[:, :, :-1].detach()))
        return qe, ke

    vh = [layer.self_attn.v_proj.register_forward_hook(
        lambda m, i, o: caps.setdefault("v", []).append(o.detach())) for layer in model.model.layers]
    ML.apply_rotary_pos_emb = grab
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        ML.apply_rotary_pos_emb = orig
        for h in vh:
            h.remove()
    per = {label(f, r): {"err": [], "bound": []} for f, r in FMTS[1:]}
    for (q, k), v in zip(caps["qk"], caps["v"]):
        B, N = 1, a.prefix
        v = v[:, :-1].view(B, N, Hk, hd).transpose(1, 2)
        q16, k16, v16 = q.half().cuda(), k.half().cuda().contiguous(), v.half().cuda().contiguous()
        G = Hq // Hk
        w = torch.softmax((q16.double() @ k16.double().repeat_interleave(G, 1).transpose(-1, -2)) * hd ** -0.5, -1)
        o = w @ v16.double().repeat_interleave(G, 1)
        for f, r in FMTS[1:]:
            kc, vc = KQ.quantize(k16, f, a.group, rot=r), KQ.quantize(v16, f, a.group, rot=r)
            got = KQ.decode_attention(q16, kc, vc, deterministic=True)
            per[label(f, r)]["err"].append(float((got.double() - o).abs().max()))
            per[label(f, r)]["bound"].append(float(KQ.error_bound(q16, kc, vc, k16, v16).max()))
    rows = {}
    for name, d in per.items():
        e, b = sorted(d["err"]), sorted(d["bound"])
        rows[name] = {"err_median": e[len(e) // 2], "err_max": e[-1], "bound_median": b[len(b) // 2],
                      "bound_max": b[-1], "bound_over_err_min": min(x / max(y, 1e-30) for x, y in zip(d["bound"], d["err"]))}
        print("layers", name, {k: round(v, 5) for k, v in rows[name].items()}, flush=True)
    out["decode_layers"] = rows

    # 2. perplexity with the cache in each format
    n, T = (int(x) for x in a.ppl.split("x"))
    ev = text_ids(tok, "test", n, T, offset=a.prefix + 1)
    ppl = {}
    for f, r in [("none", False)] + FMTS:
        def rope(q, k, cos, sin, *rest, f=f, r=r, **kw):
            qe, ke = orig(q, k, cos, sin, *rest, **kw)
            return qe, (ke if f == "none" else cache_values(ke, f, r).to(ke.dtype))

        hooks = []
        if f != "none":
            def vq(m, i, o, f=f, r=r):
                B_, L_, _ = o.shape
                x = o.view(B_, L_, Hk, hd).transpose(1, 2)
                return cache_values(x, f, r).to(o.dtype).transpose(1, 2).reshape(B_, L_, -1)
            hooks = [layer.self_attn.v_proj.register_forward_hook(vq) for layer in model.model.layers]
        ML.apply_rotary_pos_emb = rope
        try:
            with torch.no_grad():
                lg = model(input_ids=ev, use_cache=False).logits.double()
        finally:
            ML.apply_rotary_pos_emb = orig
            for h in hooks:
                h.remove()
        lp = torch.log_softmax(lg[:, :-1], -1)
        ce = float(-lp.gather(-1, ev[:, 1:, None]).mean())
        key = "unquantized (float32)" if f == "none" else label(f, r)
        ppl[key] = {"ce": ce, "ppl": float(torch.tensor(ce).exp())}
        print("ppl", key, ppl[key], flush=True)
    base = ppl["float16"]["ce"]
    for v_ in ppl.values():
        v_["delta_ce_vs_float16_cache"] = v_["ce"] - base
    out["perplexity"] = {"tokens": list(ev.shape), "rows": ppl}

    # 3. memory
    out["bytes_per_token"] = {
        "this model (32 layers, 5 KV heads of 64)": {label(f, r): 32 * KQ.bytes_per_token(f, hd, a.group, Hk)
                                                    for f, r in FMTS if not r},
        "8 KV heads of 128, one layer": {label(f, r): KQ.bytes_per_token(f, 128, a.group, 8) for f, r in FMTS if not r}}
    print(out["bytes_per_token"], flush=True)

    # 4. speed, briefly
    if a.bench:
        import triton
        import torch.nn.functional as Fn
        res = []
        for B, N in ((1, 4096), (1, 16384), (8, 4096), (8, 16384)):
            g = torch.Generator(device="cuda").manual_seed(0)
            q = torch.randn(B, 32, 1, 128, device="cuda", generator=g).half()
            k = torch.randn(B, 8, N, 128, device="cuda", generator=g).half()
            v = torch.randn(B, 8, N, 128, device="cuda", generator=g).half()
            row = {"B": B, "N": N}
            row["float16 sdpa"] = triton.testing.do_bench(
                lambda: Fn.scaled_dot_product_attention(q, k, v, enable_gqa=True), rep=50) * 1000
            for f, r in FMTS[1:]:
                kc, vc = KQ.quantize(k, f, a.group, rot=r), KQ.quantize(v, f, a.group, rot=r)
                KQ.decode_attention(q, kc, vc)
                row[label(f, r)] = triton.testing.do_bench(
                    lambda: KQ.decode_attention(q, kc, vc, verify=False), rep=50) * 1000
            res.append(row)
            print("bench us", {k_: (round(v_, 1) if isinstance(v_, float) else v_) for k_, v_ in row.items()}, flush=True)
            del q, k, v
            torch.cuda.empty_cache()
        out["bench_us"] = res
        out["gpu"] = torch.cuda.get_device_name()
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
