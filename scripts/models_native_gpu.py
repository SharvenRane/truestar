"""The native decoder on Truestar kernels, without transformers installed (for a bare GPU image).

    PYTHONPATH=src python scripts/models_native_gpu.py [--device cuda]

Writes a Hugging Face style folder (config.json and model.safetensors under the transformers
names) from random native weights for a Llama 3 style and a Gemma 2 style config, loads it with
from_pretrained twice (float64 reference interpreter; CompiledExecutor on the device in float32 and
bfloat16, which checks its first run against the float64 reference itself), and prints the worst
relative error of the logits and of four decode steps, plus the time per prefill call.
"""

import argparse
import json
import os
import tempfile
import time

import torch

from truestar import models
from truestar.interop.safetensors import save_file

CONFIGS = {
    "llama": dict(model_type="llama", vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=500000.0,
                  rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                "original_max_position_embeddings": 16}, torch_dtype="float32"),
    "gemma2": dict(model_type="gemma2", vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                   sliding_window=3, attn_logit_softcapping=5.0, final_logit_softcapping=3.0, rms_norm_eps=1e-6,
                   hidden_activation="gelu_pytorch_tanh", torch_dtype="float32"),
}


def write_folder(raw, seed=0):
    cfg = models.decoder_config(raw)
    m = models.build_model(cfg, "f32")
    p = m.init(seed=seed)
    flat = {}
    for src, dests in models.name_map(cfg).items():
        parts = []
        for d, _ in dests:
            t = p
            for k in d.split("."):
                t = t[k]
            parts.append(t)
        flat[src] = (torch.cat(parts, 0) if len(parts) > 1 else parts[0]) * 0.5
    d = tempfile.mkdtemp(prefix=f"ts_native_{raw['model_type']}_")
    save_file(flat, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(raw, f)
    return d


def rel(a, b):
    return ((a.double().cpu() - b.double().cpu()).abs().max() / b.double().abs().max()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from truestar.compiler import CompiledExecutor
    for name, raw in CONFIGS.items():
        d = write_folder(raw)
        ref = models.from_pretrained(d, dtype="f64")
        ids = torch.randint(0, 96, (2, 8), generator=torch.Generator().manual_seed(1))
        want = ref(ids)
        for prec, dt in (("float32", "f32"), ("bfloat16", "bf16")):
            lm = models.from_pretrained(d, dtype=dt, executor=CompiledExecutor(a.device, prec))
            lm.params = _to(lm.params, a.device)
            dids = ids.to(a.device)
            t0 = time.perf_counter()
            got = lm(dids)
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(10):
                got = lm(dids)
            if a.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 10 * 1e3
            cache = _to(lm.init_cache(2, 8), a.device)
            lg, cache = lm.step(dids[:, :4], torch.arange(4, device=a.device), cache)
            worst = rel(lg, want[:, :4])
            for t in range(4, 8):
                lg, cache = lm.step(dids[:, t:t + 1], torch.tensor([t], device=a.device), cache)
                worst = max(worst, rel(lg[:, 0], want[:, t]))
            print(json.dumps({"family": name, "device": a.device, "precision": prec, "prefill_rel": rel(got, want),
                              "decode_rel": worst, "first_call_s": round(first, 2), "call_ms": round(ms, 2)}))


def _to(tree, dev):
    if isinstance(tree, dict):
        return {k: _to(v, dev) for k, v in tree.items()}
    return tree.to(dev)


if __name__ == "__main__":
    main()
