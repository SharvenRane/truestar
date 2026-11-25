"""Measurements for docs/CONFORM_LLAMACPP.md: `truestar conform llamacpp` on tiny models built in code.

Nothing is downloaded. Per architecture (llama, qwen2) and seed: a sentencepiece tokenizer trained on
generated text, a decoder with random weights and random norms saved with transformers, Truestar's
float64 replay cross checked against transformers in float64, then the full check (llama.cpp's own
converter for f32, f16, q8_0, its quantizer for q4_k_m, every engine setting). Planted faults:
a rope base override in the engine (engine side), a GGUF converted from a config with another rope
theta, and one from a checkpoint with one weight scaled (conversion side).

usage: python scripts/llamacpp_conform_study.py --llama-cpp DIR --out DIR [--seeds 2] [--arch llama,qwen2]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from truestar.conform import llamacpp as L  # noqa: E402


def make_tokenizer(path: str, vocab: int = 320) -> None:
    import sentencepiece as spm

    rng = np.random.default_rng(7)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    words = ["".join(rng.choice(letters, size=rng.integers(2, 8))) for _ in range(400)]
    lines = [" ".join(rng.choice(words, size=12)) for _ in range(3000)]
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=buf, vocab_size=vocab,
                                   model_type="unigram", character_coverage=1.0, unk_id=0, bos_id=1, eos_id=2,
                                   pad_id=-1, num_threads=1, minloglevel=2)
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def make_model(out: str, arch: str, seed: int, tok: str, theta: float = 10000.0, scale_weight: str = "") -> None:
    import transformers as T

    torch.manual_seed(seed)
    common = dict(vocab_size=320, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, rms_norm_eps=1e-5,
                  rope_theta=theta, bos_token_id=1, eos_token_id=2)
    if arch == "llama":
        cfg = T.LlamaConfig(tie_word_embeddings=False, **common)
        model = T.LlamaForCausalLM(cfg)
    else:
        cfg = T.Qwen2Config(tie_word_embeddings=True, **common)
        model = T.Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1 and n.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif n.endswith(".bias"):
                p.copy_(0.05 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.08)
        if scale_weight:
            dict(model.named_parameters())[scale_weight].mul_(1.5)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    shutil.copy(tok, os.path.join(out, "tokenizer.model"))


def _float64_rope(model, theta: float, head_dim: int) -> int:
    """transformers computes the rotary angles in float32 even for a float64 model; this replaces
    the rotary module's forward with a float64 one so the rest of the model can be compared exactly."""
    n = 0
    for m in model.modules():
        if hasattr(m, "inv_freq") and type(m).__name__.endswith("RotaryEmbedding"):
            inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))

            def fwd(x, position_ids, inv=inv):
                ang = position_ids[:, :, None].to(torch.float64) * inv[None, None, :]
                emb = torch.cat([ang, ang], dim=-1)
                return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
            m.forward = fwd
            n += 1
        elif type(m).__name__.endswith("RMSNorm") and hasattr(m, "variance_epsilon"):
            # transformers' RMSNorm casts to float32 for the variance, also in a float64 model
            def nfwd(x, m=m):
                return m.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + m.variance_epsilon))
            m.forward = nfwd
            n += 1
    return n


def hf_crosscheck(d: str, prompts) -> dict:
    import transformers as T

    wt = L.load_hf_dir(d)
    out = {}
    for name in ("stock", "float64_rope_and_norm"):
        model = T.AutoModelForCausalLM.from_pretrained(d, dtype=torch.float64, attn_implementation="sdpa").eval()
        if name != "stock":
            out["patched_modules"] = _float64_rope(model, wt.cfg.theta, wt.cfg.head_dim)
        worst = 0.0
        with torch.no_grad():
            for p in prompts:
                hf = model(torch.tensor([p])).logits[0].to(torch.float64)
                ts = L.forward(wt, p)
                worst = max(worst, float((ts - hf).norm() / hf.norm()))
        out[name] = worst
    return out


def summarize(rep) -> dict:
    out = {"codes": rep.codes, "finding": rep.finding, "seconds": rep.seconds, "formats": {}}
    for f in rep.formats:
        out["formats"][f.format] = {
            "types": f.tensor_types, "codes": sorted({c for c, _ in f.issues}),
            "worst_ratio": max(r.worst_ratio for r in f.runs), "max_rel": max(r.max_rel for r in f.runs),
            "worst_abs_ratio": max(r.worst_abs_ratio for r in f.runs),
            "over": sum(len(r.over) for r in f.runs), "positions": sum(r.positions for r in f.runs),
            "greedy_divergent": sum(1 for r in f.runs for g in r.greedy if g.first_divergent is not None),
            "greedy_near_tie": sum(1 for r in f.runs for g in r.greedy if g.first_divergent is not None and g.near_tie),
            "flips": sum(r.flips for r in f.runs), "flips_near_tie": sum(r.flips_near_tie for r in f.runs),
            "settings": {k: v for k, v in f.settings.items() if k != "identical_groups"},
            "groups": f.settings.get("identical_groups"),
            "quantization": f.quantization, "conversion": f.conversion,
            "total_rel_max": max((r.total_rel or 0.0) for r in f.runs)}
    return out


def print_summary(path: str) -> None:
    """The tables of docs/CONFORM_LLAMACPP.md from a saved study.json."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    print("model format: worst/tau worst_abs/bound/10 over/positions greedy_div(near) flips(near) "
          "groups token_changes | quant rel_max top1 beyond_ties greedy_changed | total_rel_max | codes")
    for r in d["runs"]:
        print(r["model"], "hf float64 cross check", r["hf_vs_truestar_float64"])
        for fmt, x in r["clean"]["formats"].items():
            q = x["quantization"] or {}
            print(f"  {fmt:7s} {x['worst_ratio']:.3f} {x['worst_abs_ratio']:.2f} {x['over']}/{x['positions']} "
                  f"{x['greedy_divergent']}({x['greedy_near_tie']}) {x['flips']}({x['flips_near_tie']}) "
                  f"{x['settings']['groups']} {x['settings']['n_token_changes']} | {q.get('rel_max', 0):.2e} "
                  f"{q.get('top1_agreement', 0):.3f} {q.get('flips_beyond_near_ties')} {q.get('greedy_changed')} | "
                  f"{x['total_rel_max']:.2e} | {','.join(x['codes'])}")
            print("          groups:", x["groups"])
        for k, v in r.items():
            if k.startswith("fault_"):
                fx = list(v["formats"].values())[0]
                print(f"  {k}: codes {v['codes']} finding {v['finding']} worst/tau {fx['worst_ratio']:.2f} "
                      f"conversion {fx['conversion'][:1]}")


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        print_summary(sys.argv[2])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arch", default="llama,qwen2")
    ap.add_argument("--formats", default=",".join(L.FORMATS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = os.path.join(a.out, "tokenizer.model")
    if not os.path.isfile(tok):
        make_tokenizer(tok)
    build = L.find_build(a.llama_cpp)
    results = {"build": build.version, "runs": []}
    formats = a.formats.split(",")
    for arch in a.arch.split(","):
        for seed in range(a.seeds):
            tag = f"{arch}-s{seed}"
            d = os.path.join(a.out, tag)
            if not os.path.isdir(d):
                make_model(d, arch, seed, tok)
            prompts = L.default_prompts(320, bos=1)
            row = {"model": tag, "hf_vs_truestar_float64": hf_crosscheck(d, prompts)}
            t0 = time.perf_counter()
            rep = L.conform_llamacpp(d, build=build, formats=formats, workdir=d)
            row["clean"] = summarize(rep)
            with open(os.path.join(a.out, f"{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(rep.to_text())
            # engine side planted fault: the engine told to use another rope base than the file holds
            f32 = os.path.join(d, f"{tag}-f32.gguf")
            bad = L.conform_llamacpp(f32, build=build, source=d,
                                     settings=[L.Setting(4, 512, False, "f16", rope_base=500000.0),
                                               L.Setting(4, 512, False, "f16", rope_base=10100.0)])
            row["fault_engine_rope_base"] = summarize(bad)
            # conversion side planted faults: GGUF converted from a wrong config or a scaled weight
            for name, kw in (("theta", {"theta": 500000.0}),
                             ("scaled_up_l1", {"scale_weight": "model.layers.1.mlp.up_proj.weight"})):
                fd = os.path.join(a.out, f"{tag}-fault-{name}")
                if not os.path.isdir(fd):
                    make_model(fd, arch, seed, tok, **kw)
                files = L.convert(build, fd, fd, ["f32", "q8_0"])
                for fmt, path in files.items():
                    r = L.conform_llamacpp(path, build=build, source=d, settings=[L.Setting(4, 512, False)])
                    row[f"fault_{name}_{fmt}"] = summarize(r)
            row["seconds"] = time.perf_counter() - t0
            results["runs"].append(row)
            print(json.dumps({"model": tag, "hf": row["hf_vs_truestar_float64"],
                              "clean_codes": row["clean"]["codes"],
                              "engine_fault_codes": row["fault_engine_rope_base"]["codes"],
                              "seconds": round(row["seconds"], 1)}), flush=True)
    with open(os.path.join(a.out, "study.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
