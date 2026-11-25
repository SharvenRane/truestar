"""Train a small target and draft model pair for the speculative decoding benchmark.

    PYTHONPATH=src python scripts/specdec_pair.py --out DIR [--steps 1500]

Byte level (vocab 256) Llama style models (examples/llama_paged_decode.py) trained on the Python
standard library's own source files, which every Python install has, so the pair is rebuilt from the
seed on any machine and nothing is downloaded. The target is larger than the draft; both learn the
same text, so the draft's greedy guesses agree with the target often enough for a real acceptance
rate (random weights agree about 1 time in 256). Writes target.pt and draft.pt (state dicts and
configs) and a held out prompt set prompts.pt.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))

TARGET = dict(vocab=256, dim=384, heads=8, kv_heads=4, head_dim=48, layers=8, ffn=1024, block=16)
DRAFT = dict(vocab=256, dim=128, heads=4, kv_heads=2, head_dim=32, layers=2, ffn=384, block=16)


def corpus(limit=24_000_000):
    """Bytes of the standard library's .py files, sorted by path (the same on every run)."""
    lib = os.path.join(sys.base_prefix, "Lib")
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(lib) for f in fs
                   if f.endswith(".py") and "site-packages" not in d and "test" not in d.lower())
    out = bytearray()
    for f in files:
        with open(f, "rb") as h:
            out += h.read()
        if len(out) >= limit:
            break
    return bytes(out)


def train(cfg, data, steps, seed, seq=256, batch=32, lr=1e-3):
    import torch
    import llama_paged_decode as P
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True     # training only; the benchmark runs IEEE float32
    c = P.Cfg(**cfg)
    m = P.Llama(c).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(seed)
    n = len(data) - seq - 1
    z = torch.zeros(c.layers, batch, c.kv_heads, seq, c.head_dim, device="cuda")
    pos = torch.arange(seq, device="cuda")
    t = time.perf_counter()
    for i in range(steps):
        starts = torch.randint(0, n, (batch,), generator=g)
        x = torch.stack([data[s:s + seq + 1] for s in starts]).cuda()
        logits, _, _ = m.dense_step(x[:, :-1], pos, z, z)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % 250 == 0 or i == steps - 1:
            print(f"  step {i:5d} loss {loss.item():.3f} ({time.perf_counter() - t:.0f} s)", flush=True)
    return m.eval().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    import torch
    raw = corpus()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
    cut = int(len(data) * 0.98)
    train_d, held = data[:cut], data[cut:]
    print(f"corpus: {len(raw) / 1e6:.1f} MB of standard library source; held out {len(held) / 1e6:.2f} MB")
    os.makedirs(a.out, exist_ok=True)
    for name, cfg, seed in (("target", TARGET, 1), ("draft", DRAFT, 2)):
        print(f"training {name} {cfg}")
        m = train(cfg, train_d, a.steps, seed)
        torch.save({"cfg": cfg, "state": m.state_dict()}, os.path.join(a.out, f"{name}.pt"))
    g = torch.Generator().manual_seed(3)
    starts = torch.randint(0, len(held) - 64, (16,), generator=g)
    torch.save(torch.stack([held[s:s + 64] for s in starts]), os.path.join(a.out, "prompts.pt"))
    print("wrote the pair to the --out directory")


if __name__ == "__main__":
    main()
