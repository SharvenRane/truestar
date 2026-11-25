"""Decode benchmark: paged KV cache against the dense cache (both compiled by Truestar) and against
eager Hugging Face ``generate``, on one Llama config built in code (random weights, no download).

    PYTHONPATH=src python scripts/bench_decode.py [--batches 1,8] [--new 64] [--rounds 3]

Every method decodes greedily from the same prompt. Reported: milliseconds per generated token
(median over interleaved rounds, after a warm up generation that also runs Truestar's first call
checks), and whether the tokens agree: paged against dense bitwise on the logits, both against
``generate`` token by token.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import llama_paged_decode as P  # noqa: E402


def hf_model(vocab, dim, layers, heads, kv, ffn):
    import transformers

    cfg = transformers.LlamaConfig(vocab_size=vocab, hidden_size=dim, num_hidden_layers=layers,
                                   num_attention_heads=heads, num_key_value_heads=kv, intermediate_size=ffn,
                                   max_position_embeddings=4096, tie_word_embeddings=False,
                                   attn_implementation="sdpa")
    torch.manual_seed(0)
    m = transformers.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():   # larger weights than the default init, so greedy tokens vary
        for p in m.model.layers.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
    return m


def timed(fn, sync=True):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=1408)
    ap.add_argument("--block", type=int, default=16)
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    from truestar.compiler import compile_program

    hf = hf_model(a.vocab, a.dim, a.layers, a.heads, a.kv, a.ffn)
    mine = P.from_hf(hf)
    c = P.Cfg(**{**mine.c.__dict__, "block": a.block})
    mine.c = c
    width = -(-(a.prompt + a.new) // a.block)
    capacity = width * a.block
    t0 = time.perf_counter()
    dense, paged = P.import_steps(mine, capacity=capacity, n_blocks=4 * width, width=width)
    t_import = time.perf_counter() - t0
    t0 = time.perf_counter()
    cd = compile_program(dense, "cuda", "float32")
    cp = compile_program(paged, "cuda", "float32")
    t_compile = time.perf_counter() - t0
    hf = hf.cuda()
    mine = mine.cuda()
    print(f"config: vocab {a.vocab}, dim {a.dim}, layers {a.layers}, heads {a.heads}/{a.kv}, ffn {a.ffn}, "
          f"block {a.block}, prompt {a.prompt}, new tokens {a.new}, capacity {capacity}")
    print(f"import (both steps, float64 trace): {t_import:.1f} s; compile: {t_compile:.1f} s; "
          f"fallbacks dense {len(cd.report.fallbacks)}, paged {len(cp.report.fallbacks)}")
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
    print()
    print("| batch | method | ms per token | tokens per s | agreement |")
    print("|---|---|---|---|---|")
    for b in [int(x) for x in a.batches.split(",")]:
        prompt = torch.randint(0, a.vocab, (b, a.prompt), generator=torch.Generator().manual_seed(b))
        n_blocks = b * width + 3
        order = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(7)).tolist()

        def run_dense():
            return P.generate_dense(lambda *x: cd(list(x)), prompt, a.new, c, capacity, device="cuda",
                                    dtype=torch.float32)

        def run_paged():
            return P.generate_paged(lambda *x: cp(list(x)), prompt, a.new, c, n_blocks, width, order=order,
                                    device="cuda", dtype=torch.float32)

        def run_eager_dense():
            with torch.no_grad():
                return P.generate_dense(mine.dense_step, prompt, a.new, c, capacity, device="cuda",
                                        dtype=torch.float32)

        def run_hf():
            with torch.no_grad():
                out = hf.generate(prompt.cuda(), max_new_tokens=a.new, min_new_tokens=a.new, do_sample=False,
                                  use_cache=True, pad_token_id=0, eos_token_id=None)
            return out[:, a.prompt:], None

        methods = [("Truestar compiled, paged cache", run_paged), ("Truestar compiled, dense cache", run_dense),
                   ("eager PyTorch, same dense step", run_eager_dense), ("eager Hugging Face generate", run_hf)]
        results = {name: fn() for name, fn in methods}           # warm up and first call checks
        times = {name: [] for name, _ in methods}
        for _ in range(a.rounds):
            for name, fn in methods:
                dt, _ = timed(fn)
                times[name].append(dt)
        dtok, dlog = results["Truestar compiled, dense cache"]
        ptok, plog = results["Truestar compiled, paged cache"]
        htok, _ = results["eager Hugging Face generate"]
        agree = {
            "Truestar compiled, paged cache": f"logits {'bitwise equal' if torch.equal(plog, dlog) else 'differ'} to dense",
            "Truestar compiled, dense cache": f"{(dtok.cpu() == htok.cpu()).float().mean().item() * 100:.0f}% tokens equal to generate",
            "eager PyTorch, same dense step": f"{(results['eager PyTorch, same dense step'][0].cpu() == htok.cpu()).float().mean().item() * 100:.0f}% tokens equal to generate",
            "eager Hugging Face generate": "reference",
        }
        for name, _ in methods:
            med = statistics.median(times[name])
            ms = med / a.new * 1e3
            print(f"| {b} | {name} | {ms:.2f} | {b * a.new / med:.0f} | {agree[name]} |")
    print(f"\npeak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
