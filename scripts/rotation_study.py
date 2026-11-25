"""Rotations before quantization on a real pretrained model (truestar.rotate).

    PYTHONPATH=src python scripts/rotation_study.py --json results/rotation/study.json

Model: SmolLM2-360M (Llama layout: hidden 960 = 60 x 16, 32 layers, 15 query and 5 KV heads of
dim 64, SwiGLU 2560 = 20 x 128, tied embeddings) from the local Hugging Face cache, no download.
Text: WikiText-2 raw, test split for evaluation, train split for calibration.

1. Proof: the Hadamard plan (norm fold, R1, R2, online R4) is applied and proven equal to the
   original in float64 at every checked point; the three wrong placements are applied too and the
   point each one fails at is recorded.
2. A learned R1 (truestar.rotate.learned, Cayley parametrized, started at the Hadamard, trained
   for ``--steps`` steps on a layer local W4A4 objective), proven like the others.
3. Format loss (verdict b of docs/QUANT.md, float64 references only) of the formats a rotation is
   offered for (W4A4 and W8A8 per token integers, `qerror.ROTATION_FORMATS`) on: the original;
   Hadamard R1 + R2 + R4;
   Hadamard R1 + R2 (offline only);
   two random orthogonal R1 (seeds 0, 1) with Hadamard R2 + R4; the learned R1 with Hadamard
   R2 + R4. Every rotated row is compared with the original's float64 logits. The block and float
   formats (int4:g64, nvfp4, mxfp4, fp8_e4m3:channel) were measured 1.9x to 5.9x worse rotated
   (docs/ROTATION.md) and a rotated model is now refused for them (TS1024).

The float64 work runs on ``--device`` (cpu by default, ``--threads`` threads; cuda works too, a
float64 matmul on the GPU is still float64) holding at most two float64 copies of the model (2.9 GB
each): the format references are swapped into the model in place. The model is loaded in its
checkpoint dtype bfloat16, which float64 holds exactly. The learned R1 trains in float32 on
``--learn-device``.
"""

import argparse
import json
import os
import time


def load(name, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=dtype or torch.float32,
                                                 attn_implementation="eager").eval()
    return tok, model


def text_ids(tok, split, n, T, offset=0):
    import pandas as pd
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", local_files_only=True)
    df = pd.read_parquet(os.path.join(root, "wikitext-2-raw-v1", f"{split}-00000-of-00001.parquet"))
    ids = tok("".join(df["text"].tolist()[:4000]), return_tensors="pt").input_ids[0]
    ids = ids[offset: offset + n * T]
    return ids.view(n, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--learn-device", default="cpu")
    ap.add_argument("--calib", default="1x256")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eval", default="2x384")
    ap.add_argument("--specs", default="w4a4_int_rtn,w8a8_int_rtn")
    ap.add_argument("--json")
    ap.add_argument("--learned-cache", help="a .pt file: the learned R1 and its log are saved there and reused")
    a = ap.parse_args()

    import torch
    from truestar.rotate import fold as R
    from truestar.rotate import learned as LR
    from truestar.rotate import qerror as QE

    torch.set_num_threads(a.threads)
    tok, model = load(a.model, torch.bfloat16)
    dev = a.device
    n, T = (int(x) for x in a.eval.split("x"))
    ev = text_ids(tok, "test", n, T)
    calib = text_ids(tok, "train", *(int(x) for x in a.calib.split("x")))
    specs = a.specs.split(",")
    out = {"model": a.model, "eval_tokens": list(ev.shape), "specs": specs}

    # 1. proof
    t0 = time.time()
    had = R.apply(model, R.Plan(), device=dev)
    rep = R.prove(model, had, ev[:1, :128])
    del had
    out["proof"] = {"points": len(rep.points), "worst_rel": rep.worst, "tol": rep.tol,
                    "orthogonality": rep.orthogonality, "seconds": time.time() - t0}
    print("proof", out["proof"], flush=True)
    wrong = {}
    for label, plan in (("norm not folded", R.Plan(fold_norms=False)),
                        ("head rotation across RoPE", R.Plan(extra=["qk_pre_rope"])),
                        ("rotation across SiLU", R.Plan(r4=None, extra=["mlp_across_act"]))):
        r = R.prove(model, R.apply(model, plan, device=dev), ev[:1, :128], raise_on_error=False)
        p = next(x for x in r.points if x.name == r.first_bad)
        wrong[label] = {"first_bad": r.first_bad, "rel": p.rel, "suspects": r.suspects}
        print(label, wrong[label], flush=True)
    out["wrong_placements"] = wrong

    # 2. learned R1
    log = []
    t0 = time.time()
    if a.learned_cache and os.path.exists(a.learned_cache):
        saved = torch.load(a.learned_cache)
        q_learn, log, secs = saved["q"], saved["log"], saved["seconds"]
    else:
        q_learn = LR.learn_r1(model, calib, steps=a.steps, log=log, device=a.learn_device)
        secs = time.time() - t0
        if a.learned_cache:
            torch.save({"q": q_learn, "log": log, "seconds": secs}, a.learned_cache)
    out["learned"] = {"steps": a.steps, "loss_start": log[0]["loss"], "loss_end": log[-1]["loss"],
                      "seconds": secs, "log": log[:: max(1, a.steps // 20)]}
    print("learned", {k: v for k, v in out["learned"].items() if k != "log"}, flush=True)

    # 3. format loss with and without rotation
    m64 = R.to_float64(model, dev)
    ev = ev.to(dev)
    ref = QE.logits(m64, ev)
    variants = [("none", None), ("hadamard R1 R2 R4", R.Plan()),
                ("hadamard R1 R2 (offline only)", R.Plan(r4=None))]
    for seed in (0, 1):
        g = torch.Generator().manual_seed(seed)
        qr, _ = torch.linalg.qr(torch.randn(model.config.hidden_size, model.config.hidden_size,
                                            dtype=torch.float64, generator=g))
        variants.append((f"random R1 (seed {seed}) + hadamard R2 R4", R.Plan(r1=qr)))
    variants.append(("learned R1 + hadamard R2 R4", R.Plan(r1=q_learn)))
    rows = []
    for label, plan in variants:
        t0 = time.time()
        if plan is None:
            mm = m64
            proof = None
        else:
            rot = R.apply(model, plan, device=dev)
            proof = R.prove(model, rot, ev[:1, :128]).worst
            mm = rot.model
            R.norms64(mm)
        res = QE.evaluate(mm, ev, specs, ref_logits=ref, layer_ids=ev[:1, :128])
        for f in res["formats"].values():
            f.pop("layers")
        rows.append({"variant": label, "proof_worst_rel": proof, **res, "seconds": time.time() - t0})
        print(label, json.dumps({s: {k: round(v, 5) for k, v in f.items()} for s, f in res["formats"].items()}),
              "act max/rms", round(res["act_max_over_rms"], 2), flush=True)
        if plan is None:
            del m64
        del mm
        if plan is not None:
            del rot
        torch.cuda.empty_cache()
    out["variants"] = rows
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
