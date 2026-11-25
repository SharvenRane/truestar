"""truestar serve on a GPU: the guarantee check, cross executor agreement, and tokens per second.

    PYTHONPATH=src python scripts/bench_serve.py [--batches 1,8,32] [--prompt 32] [--new 64] [--rounds 3]
                                                 [--hf] [--json out.json]

1. ``serve --check`` on cuda (tiny preset): bitwise alone, batched with 7 others, prefix cache on,
   chunked prefill, all combined; float64 agreement. The same with two planted departures, to show what
   the design rules out: the compiled step without deterministic mode (split K allowed), and attention
   sized to the batch's widest sequence instead of the fixed table width.
2. cuda against the numba cpu executor: the same requests (greedy and keyed sampling), tokens and
   logprobs.
3. Throughput on the ``bench`` preset (vocab 8192, width 512, 4 layers, 8/2 heads, random weights): B
   requests of ``--prompt`` random tokens, ``--new`` greedy tokens each, all submitted at once.
   * Truestar serve, compiled cuda step (float32, deterministic).
   * Truestar serve's scheduler with the same step in eager PyTorch (float32).
   * eager PyTorch generate: a static batch loop over a dense KV cache (prefill, then one step per
     token), the way ``generate`` runs; float32, same weights.
   * Hugging Face ``generate`` with the same weights (``--hf``; installs transformers when missing).
   Each row also says whether request 0 got the same bits as when it ran alone (batch 1).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time

import numpy as np
import torch


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def guarantee(out):
    from truestar.serve.check import _requests, check_invariance, drive
    from truestar.serve.engine import Engine, StepRunner
    from truestar.serve.model import TinyLM, config

    m = TinyLM(config("tiny"))
    cuda = StepRunner(m, "cuda")
    rep = check_invariance(m, cuda, StepRunner(m, "reference"))
    out["check_cuda"] = rep
    print("check cuda:", rep["codes"], "float64:", {k: rep["float64"][k] for k in
                                                    ("tokens_equal", "float32_logprob_max_error")}, flush=True)
    nd = StepRunner(m, "cuda", deterministic=False)
    rep2 = check_invariance(m, nd)
    out["check_cuda_not_deterministic"] = rep2
    print("check cuda, deterministic=False:", rep2["codes"],
          [(r["scenario"], r["request"], r["first_difference"]) for r in rep2["rows"] if not r["bitwise"]], flush=True)
    import truestar.serve.check as C

    class Narrow(Engine):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._batch_width = True

    C.Engine = Narrow
    try:
        rep3 = check_invariance(m, cuda)
    finally:
        C.Engine = Engine
    out["check_cuda_batch_width"] = rep3
    print("check cuda, attention width from the batch:", rep3["codes"],
          [(r["scenario"], r["request"], r["first_difference"]) for r in rep3["rows"] if not r["bitwise"]], flush=True)
    cpu = StepRunner(m, "cpu")
    targets, _, _ = _requests(m.c)
    cross = []
    for t in targets:
        a = drive(Engine(m, runner=cuda), [(0, t)])[0]
        b = drive(Engine(m, runner=cpu), [(0, t)])[0]
        cross.append({"request": t.rid, "tokens_equal": a.out == b.out, "logprobs_bitwise": a.logprobs == b.logprobs,
                      "logprob_max_diff": max(abs(x - y) for x, y in zip(a.logprobs, b.logprobs))})
    out["cuda_vs_cpu"] = cross
    print("cuda vs cpu:", cross, flush=True)


def eager_generate(model, prompt, new):
    """Static batch greedy loop over a dense cache: what ``generate`` does, in eager float32."""
    c = model.c
    b, t0 = prompt.shape
    kc = torch.zeros(c.layers, b, c.kv_heads, t0 + new, c.head_dim, device="cuda")
    vc = torch.zeros_like(kc)
    toks, lps = [], []
    with torch.no_grad():
        lg = model.dense_step(prompt, 0, kc, vc)[:, -1]
        for i in range(new):
            lp = torch.log_softmax(lg.double(), -1)
            nxt = lg.argmax(-1)
            toks.append(nxt)
            lps.append(lp.gather(1, nxt[:, None])[:, 0])
            if i + 1 < new:
                lg = model.dense_step(nxt[:, None], t0 + i, kc, vc)[:, -1]
    return torch.stack(toks, 1).cpu(), torch.stack(lps, 1).cpu()


def throughput(out, batches, prompt_len, new, rounds, hf):
    import copy

    from truestar.serve.engine import Engine, Request, StepRunner
    from truestar.serve.model import TinyLM, config, to_hf

    c = config("bench")
    m = TinyLM(c)
    t0 = time.perf_counter()
    comp = StepRunner(m, "cuda")
    print(f"bench preset: import + compile {time.perf_counter() - t0:.1f} s, fallbacks {comp.fallbacks}", flush=True)
    eag = StepRunner(m, "eager-cuda")
    m32 = copy.deepcopy(m).float().cuda()
    hfm = None
    if hf:
        try:
            import transformers  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.48.0"], check=True)
        hfm = to_hf(m).float().cuda()
    g = np.random.Generator(np.random.PCG64(5))
    prompts = [[int(x) for x in g.integers(0, 256, prompt_len)] for _ in range(max(batches))]

    def serve_run(runner, b):
        eng = Engine(m, runner=runner, max_batch=b, prefix_cache=False, max_rows=512)
        seqs = [eng.submit(Request(prompts[i], max_tokens=new)) for i in range(b)]
        sync()
        t = time.perf_counter()
        eng.run_until_idle()
        sync()
        return time.perf_counter() - t, [s.out for s in seqs], [s.logprobs for s in seqs], eng.steps

    def eager_run(b):
        p = torch.tensor(prompts[:b], device="cuda")
        sync()
        t = time.perf_counter()
        toks, lps = eager_generate(m32, p, new)
        sync()
        return time.perf_counter() - t, toks.tolist(), lps.tolist(), new

    def hf_run(b):
        p = torch.tensor(prompts[:b], device="cuda")
        sync()
        t = time.perf_counter()
        with torch.no_grad():
            o = hfm.generate(p, max_new_tokens=new, min_new_tokens=new, do_sample=False, use_cache=True,
                             pad_token_id=0, eos_token_id=None, output_scores=True, return_dict_in_generate=True)
        sync()
        dt = time.perf_counter() - t
        toks = o.sequences[:, prompt_len:]
        lps = torch.stack([torch.log_softmax(s.double(), -1).gather(1, toks[:, i:i + 1])[:, 0]
                           for i, s in enumerate(o.scores)], 1)
        return dt, toks.tolist(), lps.tolist(), new

    # warm up: every row bucket of the compiled step, and each method once
    Engine(m, runner=comp, max_rows=512).warmup()
    methods = [("truestar serve (compiled cuda, deterministic)", lambda b: serve_run(comp, b)),
               ("truestar serve scheduler, eager PyTorch step", lambda b: serve_run(eag, b)),
               ("eager PyTorch generate (static batch, dense cache)", eager_run)]
    if hfm is not None:
        methods.append(("Hugging Face generate (sdpa)", hf_run))
    for _, fn in methods:
        fn(2)
    rows = []
    alone = {}
    # float64: the compiled server's greedy tokens against a teacher forced float64 forward
    _, toks1, lps1, _ = serve_run(comp, 1)
    with torch.no_grad():
        full = m.cuda().full_logits(torch.tensor([prompts[0] + toks1[0][:-1]], device="cuda"))[0, prompt_len - 1:]
    ref_tok = full.argmax(-1).tolist()
    ref_lp = torch.log_softmax(full, -1)
    first = next((i for i, (a, b) in enumerate(zip(toks1[0], ref_tok)) if a != b), None)
    n = first if first is not None else new
    f64 = {"tokens_equal": first is None, "first_difference": first,
           "logprob_max_error": max(abs(ref_lp[i, t].item() - lps1[0][i]) for i, t in enumerate(toks1[0][:n]))}
    if first is not None:
        r = full[first]
        f64["tie_gap"] = abs(r[toks1[0][first]].item() - r[ref_tok[first]].item()) / r.abs().max().item()
    out["bench_float64"] = f64
    print("bench preset vs float64:", f64, flush=True)
    m.cpu()
    for b in batches:
        for name, fn in methods:
            ts = []
            for _ in range(rounds):
                dt, toks, lps, steps = fn(b)
                ts.append(dt)
            med = statistics.median(ts)
            key = (toks[0], lps[0])
            if b == batches[0]:
                alone[name] = key
            same = key == alone[name]
            tok_same = key[0] == alone[name][0]
            row = {"batch": b, "method": name, "seconds": med, "tokens_per_s": b * new / med,
                   "ms_per_step": med / steps * 1e3, "request0_bitwise_vs_alone": same,
                   "request0_tokens_vs_alone": tok_same, "steps": steps}
            rows.append(row)
            print(f"| {b} | {name} | {row['tokens_per_s']:.0f} | {row['ms_per_step']:.2f} | "
                  f"{'bitwise' if same else ('tokens equal, logprobs differ' if tok_same else 'TOKENS DIFFER')} |",
                  flush=True)
    out["throughput"] = rows
    out["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "prompt": prompt_len, "new": new,
                  "rounds": rounds}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    out = {}
    if not a.skip_check:
        guarantee(out)
    throughput(out, [int(x) for x in a.batches.split(",")], a.prompt, a.new, a.rounds, a.hf)
    text = json.dumps(out, indent=1, default=str)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(text)
    print("JSON_BEGIN")
    print(json.dumps(out, default=str))
    print("JSON_END")


if __name__ == "__main__":
    main()
