"""safetensors load time and load checks on real Hugging Face checkpoints.

Run: PYTHONPATH=src python scripts/safetensors_bench.py [--scratch DIR]

Needs the cached Qwen2.5-0.5B (988 MB, bfloat16) and GPT2 (548 MB) files under
~/.cache/huggingface/hub, the safetensors package (for comparison only) and transformers.
Writes a torch.save copy of the Qwen weights into --scratch for the torch.load rows.
All timings are warm (each file read once before timing; the OS file cache is not dropped).
Prints one JSON document.
"""

import argparse
import glob
import json
import os
import statistics
import time

import torch

from truestar.interop import SafetensorsError, load_file, load_weights, read_header, save_file, save_weights


def cached(repo):
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo}/snapshots/*/model.safetensors"))
    return hits[0] if hits else None


def touch(d):
    """Read every byte (a byte sum per tensor), so lazy and eager loaders do the same work."""
    s = 0
    for t in d.values():
        if t.numel():
            s += int(t.reshape(-1).view(torch.uint8).sum(dtype=torch.int64))
    return s


def med(fn, n=5):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return round(statistics.median(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("TEMP", "."))
    args = ap.parse_args()
    import safetensors
    import safetensors.torch as st
    doc = {"torch": torch.__version__, "safetensors": safetensors.__version__, "timings_s": {}, "checks": {}}
    qwen = cached("Qwen--Qwen2.5-0.5B")
    if qwen:
        size = os.path.getsize(qwen)
        doc["qwen_file_bytes"] = size
        pt = os.path.join(args.scratch, "qwen_bench.pt")
        if not os.path.exists(pt):
            torch.save(st.load_file(qwen), pt)
        T = doc["timings_s"]
        T["header validation only (truestar read_header)"] = med(lambda: read_header(qwen))
        T["truestar load_file (mmap, lazy)"] = med(lambda: load_file(qwen))
        T["truestar load_file mmap + read every byte"] = med(lambda: touch(load_file(qwen)))
        T["truestar load_file mmap=False + read every byte"] = med(lambda: touch(load_file(qwen, mmap=False)))
        T["safetensors.torch.load_file + read every byte"] = med(lambda: touch(st.load_file(qwen)))
        T["torch.load weights_only + read every byte"] = med(lambda: touch(torch.load(pt, weights_only=True)))
        T["torch.load mmap + read every byte"] = med(lambda: touch(torch.load(pt, weights_only=True, mmap=True)))
        if torch.cuda.is_available():
            def cu(fn):
                def run():
                    fn()
                    torch.cuda.synchronize()
                return run
            T["truestar load_file device=cuda"] = med(cu(lambda: load_file(qwen, device="cuda")), 3)
            T["safetensors load_file device=cuda"] = med(cu(lambda: st.load_file(qwen, device="cuda")), 3)
        a, b = load_file(qwen), st.load_file(qwen)
        doc["checks"]["qwen tensors bitwise equal to the package"] = all(torch.equal(a[k], b[k]) for k in b) \
            and set(a) == set(b)
        doc["checks"]["qwen tensor count"] = len(a)
    gpt2 = cached("openai-community--gpt2")
    if gpt2:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config.from_pretrained(os.path.dirname(gpt2))
        torch.manual_seed(0)
        model = GPT2LMHeadModel(cfg).eval()
        hf_names = set(model.state_dict())
        # the hub file stores names without the "transformer." prefix and no lm_head (tied)
        raw = load_file(gpt2)
        sd = {"transformer." + k: v for k, v in raw.items() if "transformer." + k in hf_names}
        doc["checks"]["gpt2 hub tensors the module has no slot for"] = sorted(k for k in raw if "transformer." + k not in hf_names)[:3]
        renamed = os.path.join(args.scratch, "gpt2_renamed.safetensors")
        ids = torch.tensor([[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]])
        t0 = time.perf_counter()
        save_weights(renamed, sd, check=model_with(cfg, sd), check_inputs=(ids,))
        doc["timings_s"]["gpt2 save_weights with reference record"] = round(time.perf_counter() - t0, 3)
        m = GPT2LMHeadModel(cfg).eval()
        t0 = time.perf_counter()
        res = load_weights(renamed, into=m)
        doc["timings_s"]["gpt2 load_weights into GPT2LMHeadModel with the record rerun"] = round(time.perf_counter() - t0, 3)
        doc["checks"]["gpt2 tied names filled from the file"] = res.tied
        doc["checks"]["gpt2 record error per output"] = [float(f"{e:.2e}") for e in res.check_errors]
        ref = GPT2LMHeadModel.from_pretrained(os.path.dirname(gpt2)).eval()
        with torch.no_grad():
            doc["checks"]["gpt2 logits vs from_pretrained, max abs diff"] = float((m(ids).logits - ref(ids).logits).abs().max())
        # the same square weight stored transposed: shapes still match
        bad = dict(sd)
        k = "transformer.h.0.attn.c_proj.weight"
        bad[k] = sd[k].T.contiguous()
        badpath = os.path.join(args.scratch, "gpt2_bad.safetensors")
        save_file(bad, badpath, read_header(renamed)[1])
        m2 = GPT2LMHeadModel(cfg).eval()
        m2.load_state_dict(load_file(badpath), strict=False)
        doc["checks"]["transposed c_proj, torch load_state_dict"] = "accepted (strict=False only for the tied lm_head)"
        try:
            load_weights(badpath, into=GPT2LMHeadModel(cfg).eval())
            doc["checks"]["transposed c_proj, truestar load_weights"] = "accepted"
        except SafetensorsError as e:
            doc["checks"]["transposed c_proj, truestar load_weights"] = f"refused: {e.problems[0]}"
        del res
        import gc
        gc.collect()
        for p in (renamed, badpath):
            try:
                os.remove(p)
            except PermissionError:   # Windows keeps a file open while a memory map of it lives
                pass
    print(json.dumps(doc, indent=2))


def model_with(cfg, sd):
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel(cfg).eval()
    m.load_state_dict(sd, strict=False)
    return m


if __name__ == "__main__":
    main()
