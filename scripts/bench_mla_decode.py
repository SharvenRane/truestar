"""MLA decode on one GPU: Truestar compiled steps against eager PyTorch, cache size, determinism.

Run on a CUDA machine with transformers 5.x: ``PYTHONPATH=src python scripts/bench_mla_decode.py``

One small DeepSeek V3 shaped config (dense layers, random weights scaled so greedy tokens vary),
float32, TF32 off on every path. Paths, all greedy, same weights (``truestar.mla.from_hf``):

* ``hf``: ``DeepseekV3ForCausalLM`` eager (sdpa), one forward per token with its own dynamic cache.
* ``eager expand`` and ``eager absorb``: the ``MLAModel`` step module run by PyTorch eagerly.
* ``truestar expand`` and ``truestar absorb``: the same step imported and compiled
  (``compile_steps``); prefill always runs the expanded step.

Reported: import and compile time, the float64 absorption proof, the first call checks, cache
bytes (latent against a per head K and V cache), milliseconds per generated token (median over
interleaved rounds after a warm up, the decode loop only), token agreement with the eager float64
model, and the bits of two separate compiles of the absorbed decode.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import sys
import time

import torch

from truestar.mla import MLAConfig, compile_steps, from_hf, generate, prove_absorption

CFG = dict(vocab_size=8192, hidden_size=512, intermediate_size=1408, moe_intermediate_size=256, num_hidden_layers=4,
           num_attention_heads=8, num_key_value_heads=8, q_lora_rank=192, kv_lora_rank=128, qk_nope_head_dim=32,
           qk_rope_head_dim=16, v_head_dim=32, first_k_dense_replace=4, n_routed_experts=4, num_experts_per_tok=2,
           n_group=1, topk_group=1, max_position_embeddings=1024, rope_interleave=True)


def build_hf(seed=0):
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.DeepseekV3Config(**CFG)
    cfg._attn_implementation = "sdpa"
    hf = transformers.DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
            else:
                p.normal_(1.0, 0.2)
    return hf


def hf_loop(hf, prompt, steps):
    from transformers import DynamicCache
    kw = "cache" if "cache" in inspect.signature(hf.forward).parameters else "past_key_values"
    cache = DynamicCache(config=hf.config)
    with torch.no_grad():
        out = hf(input_ids=prompt, use_cache=True, **{kw: cache})
        toks = []
        for _ in range(steps):
            nxt = out.logits[:, -1].argmax(-1)
            toks.append(nxt)
            out = hf(input_ids=nxt[:, None], use_cache=True, **{kw: cache})
    return (torch.stack(toks, 1) if toks else None), cache


def cache_bytes(cache) -> int:
    seen, n = set(), 0
    for layer in getattr(cache, "layers", []):
        for v in vars(layer).values():
            if isinstance(v, torch.Tensor) and v.data_ptr() not in seen:
                seen.add(v.data_ptr())
                n += v.numel() * v.element_size()
    return n


class TorchGraphStep:
    """The eager step module captured once into a torch.cuda.CUDAGraph (static buffers, the cache
    copied back in place): eager PyTorch without its per kernel host cost, for a fair comparison."""

    def __init__(self, model, mode):
        self.model, self.mode, self.g = model, mode, None

    def __call__(self, ids, pos, cc, rc):
        if self.g is None:
            self.ids, self.pos, self.cc, self.rc = ids.clone(), pos.clone(), cc.clone(), rc.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._body()
            torch.cuda.current_stream().wait_stream(side)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.logits = self._body()
        if cc is not self.cc:
            self.cc.copy_(cc)
            self.rc.copy_(rc)
        self.ids.copy_(ids)
        self.pos.copy_(pos)
        self.g.replay()
        return self.logits, self.cc, self.rc

    def _body(self):
        logits, cc, rc = self.model.step(self.ids, self.pos, self.cc, self.rc, self.mode)
        self.cc.copy_(cc)
        self.rc.copy_(rc)
        return logits


class GraphedDecode:
    """``cp.graphed(update=...)`` binds the cache by address: every call passes the same two cache
    tensors, so a new prompt's cache is copied into them once (as TorchGraphStep does)."""

    def __init__(self, cp):
        self.gp, self.cc, self.rc = cp.graphed(update={2: 1, 3: 2}), None, None

    def __call__(self, ids, pos, cc, rc):
        if self.cc is None:
            self.cc, self.rc = cc.clone(), rc.clone()
        elif cc is not self.cc:
            self.cc.copy_(cc)
            self.rc.copy_(rc)
        return self.gp([ids, pos, self.cc, self.rc])


def timed(fn, sync=True):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def profile(steps, e32, c, dev, n=16):
    """GPU kernel time and launches per decode step (torch.profiler), and the host functions of the
    compiled absorbed step (cProfile), batch 1."""
    import cProfile
    import io
    import pstats

    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprof

    prompt = torch.randint(0, c.vocab, (1, 64), generator=torch.Generator().manual_seed(3)).to(dev)
    _, cc, rc = (lambda o: (o[0], o[1], o[2]))(steps["absorb"].prefill([prompt, torch.arange(64, device=dev),
                                                                        *e32.empty_cache(1, 192, torch.float32, dev)]))
    ids, pos = prompt[:, -1:], torch.tensor([64], device=dev)
    fns = {"eager absorb": lambda: e32.step(ids, pos, cc, rc, "absorb"),
           "eager expand": lambda: e32.step(ids, pos, cc, rc, "expand"),
           "truestar absorb": lambda: steps["absorb"].decode([ids, pos, cc, rc]),
           "truestar expand": lambda: steps["expand"].decode([ids, pos, cc, rc])}
    out = {}
    with torch.no_grad():
        for k, f in fns.items():
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                for _ in range(n):
                    f()
                torch.cuda.synchronize()
            ev = [e for e in pr.key_averages() if e.device_type.name == "CUDA"]
            gpu_us = sum(e.self_device_time_total for e in ev) / n
            launches = sum(e.count for e in ev) / n
            t = time.perf_counter()
            for _ in range(n):
                f()
            torch.cuda.synchronize()
            out[k] = {"gpu_us_per_step": round(gpu_us, 1), "kernels_per_step": launches,
                      "wall_us_per_step": round(1e6 * (time.perf_counter() - t) / n, 1)}
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(n):
            fns["truestar absorb"]()
        torch.cuda.synchronize()
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(14)
        out["cprofile truestar absorb (16 steps)"] = buf.getvalue().splitlines()[-24:]
    return out


def determinism():
    """Hashes of the absorbed decode's tokens and logits from two separate compiles in this process,
    in the default mode and with ``deterministic=True``; run twice to compare processes."""
    torch.backends.cuda.matmul.allow_tf32 = False
    hf = build_hf()
    model = from_hf(hf)
    c = model.c
    prompt = torch.randint(0, c.vocab, (8, 64), generator=torch.Generator().manual_seed(7)).to("cuda")
    w = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()[:16]
    out = {"gpu": torch.cuda.get_device_name(), "weights_sha256": w,
           "prompt_sha256": hashlib.sha256(prompt.cpu().numpy().tobytes()).hexdigest()[:16]}
    for det, mode in ((None, "absorb"), (True, "absorb"), (None, "fused"), (True, "fused")):
        digests = []
        for _ in range(2):
            st = compile_steps(model, "cuda", "float32", decode=mode, deterministic=det)
            toks, logits = st.generate(prompt, 64, 192, c, device="cuda")
            digests.append(hashlib.sha256(toks.cpu().numpy().tobytes() + logits.cpu().numpy().tobytes()).hexdigest()[:16])
        out["%s deterministic=%s" % (mode, det)] = digests
    print(json.dumps(out, indent=1))


def main():
    if "--determinism" in sys.argv:
        determinism()
        return
    if "--profile" in sys.argv:
        torch.backends.cuda.matmul.allow_tf32 = False
        hf = build_hf()
        model = from_hf(hf)
        st = {mode: compile_steps(model, "cuda", "float32", decode=mode) for mode in ("expand", "absorb")}
        prompt = torch.randint(0, model.c.vocab, (1, 64), generator=torch.Generator().manual_seed(3)).to("cuda")
        for s in st.values():
            s.generate(prompt, 2, 192, model.c, device="cuda")     # first call checks
        print(json.dumps(profile(st, from_hf(hf).to("cuda"), model.c, "cuda"), indent=1))
        return
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    import transformers
    res["transformers"] = transformers.__version__
    hf = build_hf()
    model = from_hf(hf)
    c: MLAConfig = model.c
    res["config"] = CFG
    prompt_len, new, capacity = 64, 64, 192
    gen = torch.Generator().manual_seed(1)

    # float64 proof and reference tokens (eager, on the GPU in float64)
    m64 = from_hf(hf).double().to(dev)
    t = time.perf_counter()
    proof = prove_absorption(m64, torch.randint(0, c.vocab, (2, 16), generator=gen).to(dev), steps=4)
    res["proof"] = {"message": proof.message(), "seconds": round(time.perf_counter() - t, 2)}

    t = time.perf_counter()
    steps = {mode: compile_steps(model, dev, "float32", prefill="expand", decode=mode)
             for mode in ("expand", "absorb", "fused")}
    res["import_and_compile_seconds"] = round(time.perf_counter() - t, 1)
    hf32 = hf.to(dev)
    e32 = from_hf(hf).to(dev)

    tg, graphed = {}, {}
    for batch in (1, 8):
        prompt = torch.randint(0, c.vocab, (batch, prompt_len), generator=gen).to(dev)
        with torch.no_grad():
            ref, _ = generate(lambda *a: m64.step(*a, mode="expand"), c, prompt, new, capacity, device=dev)
        paths = {
            "hf": lambda: hf_loop(hf32, prompt, new)[0],
            "eager expand": lambda: generate(lambda *a: e32.step(*a, mode="expand"), c, prompt, new, capacity,
                                             device=dev, dtype=torch.float32)[0],
            "eager absorb": lambda: generate(lambda *a: e32.step(*a, mode="expand"), c, prompt, new, capacity,
                                             decode=lambda *a: e32.step(*a, mode="absorb"),
                                             device=dev, dtype=torch.float32)[0],
            "truestar expand": lambda: steps["expand"].generate(prompt, new, capacity, c, device=dev)[0],
            "truestar absorb": lambda: steps["absorb"].generate(prompt, new, capacity, c, device=dev)[0],
            "truestar fused": lambda: steps["fused"].generate(prompt, new, capacity, c, device=dev)[0],
            "eager absorb, torch cuda graph": lambda: generate(
                lambda *a: e32.step(*a, mode="expand"), c, prompt, new, capacity, decode=tg[(batch, "absorb")],
                device=dev, dtype=torch.float32)[0],
            "truestar absorb, graphed": lambda: generate(
                lambda *a: steps["absorb"].prefill(list(a)), c, prompt, new, capacity,
                decode=graphed[(batch, "absorb")], device=dev, dtype=torch.float32)[0],
            "truestar expand, graphed": lambda: generate(
                lambda *a: steps["expand"].prefill(list(a)), c, prompt, new, capacity,
                decode=graphed[(batch, "expand")], device=dev, dtype=torch.float32)[0],
            "truestar fused, graphed": lambda: generate(
                lambda *a: steps["fused"].prefill(list(a)), c, prompt, new, capacity,
                decode=graphed[(batch, "fused")], device=dev, dtype=torch.float32)[0],
        }
        for mode in ("absorb", "expand", "fused"):
            if mode != "fused":
                tg[(batch, mode)] = TorchGraphStep(e32, mode)
            graphed[(batch, mode)] = GraphedDecode(steps[mode].decode)
        with torch.no_grad():
            first = {k: timed(f)[0] for k, f in paths.items()}       # warm up and first call checks
            # prefill alone, to take it out of the per token time
            pre = {k: [] for k in paths}
            times = {k: [] for k in paths}
            for _ in range(rounds):
                for k, f in paths.items():
                    _, total = timed(f)
                    times[k].append(total)
        # prefill time per path: one step with the prompt only
        pre_fns = {
            "hf": lambda: hf_loop(hf32, prompt, 0),
            "eager expand": lambda: generate(lambda *a: e32.step(*a, mode="expand"), c, prompt, 1, capacity,
                                             device=dev, dtype=torch.float32),
            "eager absorb": lambda: generate(lambda *a: e32.step(*a, mode="expand"), c, prompt, 1, capacity,
                                             device=dev, dtype=torch.float32),
            "truestar expand": lambda: steps["expand"].generate(prompt, 1, capacity, c, device=dev),
            "truestar absorb": lambda: steps["absorb"].generate(prompt, 1, capacity, c, device=dev),
            "truestar fused": lambda: steps["fused"].generate(prompt, 1, capacity, c, device=dev),
        }
        pre_fns["eager absorb, torch cuda graph"] = pre_fns["eager expand"]
        pre_fns["truestar absorb, graphed"] = pre_fns["truestar absorb"]
        pre_fns["truestar expand, graphed"] = pre_fns["truestar expand"]
        pre_fns["truestar fused, graphed"] = pre_fns["truestar fused"]
        with torch.no_grad():
            for k, f in pre_fns.items():
                timed(f)
                pre[k] = statistics.median(timed(f)[1] for _ in range(rounds))
        row = {}
        for k in paths:
            decode = statistics.median(times[k]) - pre[k]
            row[k] = {"ms_per_token": round(1000 * decode / (new - 1), 3),
                      "rounds_ms_per_token": [round(1000 * (x - pre[k]) / (new - 1), 3) for x in times[k]],
                      "tokens_equal_float64": round(float((first[k].cpu() == ref.cpu()).float().mean()), 4)}
        res[f"batch {batch}"] = row
        lat, full = c.cache_bytes(batch, capacity, 4)
        _, hc = hf_loop(hf32, prompt, new)
        res[f"batch {batch} cache bytes"] = {"latent (truestar, capacity %d)" % capacity: lat,
                                             "per head K and V (capacity %d)" % capacity: full,
                                             "ratio": round(full / lat, 2),
                                             "hf dynamic cache after %d tokens" % (prompt_len + new): cache_bytes(hc)}

    # bitwise determinism: two separate compiles of the absorbed decode, same prompt
    prompt = torch.randint(0, c.vocab, (8, prompt_len), generator=torch.Generator().manual_seed(7)).to(dev)
    digests = []
    for _ in range(2):
        st = compile_steps(model, dev, "float32", decode="absorb")
        toks, logits = st.generate(prompt, new, capacity, c, device=dev)
        digests.append(hashlib.sha256(toks.cpu().numpy().tobytes() + logits.cpu().numpy().tobytes()).hexdigest()[:16])
    res["determinism"] = {"two_compiles_sha256": digests, "bitwise_equal": digests[0] == digests[1]}
    res["fallbacks"] = {m: len(s.decode.report.fallbacks) + len(s.prefill.report.fallbacks) for m, s in steps.items()}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
