"""Training steps under default verification: the fair benchmark's 12 training configs.

    PYTHONPATH=src python scripts/trainverify_bench.py [--configs mlp:adamw:plain,...] [--json out.jsonl]

The workloads and the Truestar calls of the fair benchmark (research/fairbench: fb_models.py and
fb_torch.py, copied here so this script stands alone):

* ``mlp``: Linear 1024, 4096, 4096, 1024 with tanh GELU, batch 512, cross entropy over 1024 classes;
* ``tel``: nn.TransformerEncoderLayer d 512, 8 heads, ffn 2048, ReLU, post norm, batch 32 by 256
  tokens, MSE to a fixed target;
* ``resnet18``: torchvision resnet18 with non trivial frozen BN statistics, batch 32 by 3 by 224 by
  224, cross entropy over 1000 classes;

each with AdamW (lr 1e-3, betas 0.9 and 0.999, eps 1e-8, weight decay 0.01) or SGD (lr 1e-2,
momentum 0.9). The model is imported in float32 with static shapes, the step built by
``T.build_train_step`` and compiled by ``compile_program(st.program, "cuda", "float32")`` with
default verification (no accept_ill_conditioned). ``plain`` calls ``cp(inputs, update=upd)`` (the
default: replayed from an automatic CUDA graph from the second step), ``off`` the same with
``graphs="off"`` (every step an ordinary call), ``graph`` calls ``cp.graphed(update=upd)(inputs)``, ``upd`` mapping every carried input (from input
3) to its output, as the benchmark does. Each config runs in its own process.

Per config: ``passed`` (4 steps ran, the first verified) or the first lines of the refusal; with
``--time``, the step time (median of 20 steps after 5 warm up, CUDA events) for Truestar and for
eager PyTorch (the same model, its loss, torch.optim with fused=True).
"""

import argparse
import json
import os
import subprocess
import sys

CONFIGS = [f"{m}:{o}:{g}" for m in ("mlp", "tel", "resnet18") for o in ("adamw", "sgd") for g in ("plain", "graph")]
# also: "off" (graphs="off", every step ordinary) and the "llama" workload (not in the fair benchmark)
LR, BETAS, EPS, WD = 1e-3, (0.9, 0.999), 1e-8, 0.01
SGD_LR, SGD_MOM = 1e-2, 0.9


def _llama(dim, depth, heads, vocab, seq):
    """A Llama style decoder written here (RMSNorm, rotary positions, causal attention, SwiGLU), its
    logits flattened to (tokens, vocab) for the cross entropy."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    hd = dim // heads
    ffn = 8 * dim // 3                     # SwiGLU width 2/3 of 4 dim

    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    ang = torch.arange(seq, dtype=torch.float64)[:, None] * inv[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1).float()
    sin = torch.cat([ang.sin(), ang.sin()], -1).float()
    mask = torch.full((seq, seq), float("-inf")).triu(1)

    def rope(x, cos, sin):
        half = x.shape[-1] // 2
        rot = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = RMSNorm(), RMSNorm()
            self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=False) for _ in range(4))
            self.gate, self.up = nn.Linear(dim, ffn, bias=False), nn.Linear(dim, ffn, bias=False)
            self.down = nn.Linear(ffn, dim, bias=False)
            self.register_buffer("cos", cos.clone())
            self.register_buffer("sin", sin.clone())
            self.register_buffer("mask", mask.clone())

        def forward(self, x):
            b, s, _ = x.shape
            h = self.n1(x)
            q, k, v = (t(h).reshape(b, s, heads, hd).transpose(1, 2) for t in (self.q, self.k, self.v))
            q, k = rope(q, self.cos, self.sin), rope(k, self.cos, self.sin)
            a = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5 + self.mask, -1) @ v
            x = x + self.o(a.transpose(1, 2).reshape(b, s, dim))
            h = self.n2(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    class Llama(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.norm = RMSNorm()
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, t):
            x = self.emb(t)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x)).reshape(-1, vocab)
    return Llama()


def build(workload):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(1234)
    torch.manual_seed(0)
    if workload == "mlp":
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2, self.fc3 = nn.Linear(1024, 4096), nn.Linear(4096, 4096), nn.Linear(4096, 1024)

            def forward(self, x):
                x = F.gelu(self.fc1(x), approximate="tanh")
                x = F.gelu(self.fc2(x), approximate="tanh")
                return self.fc3(x)
        m = MLP()
        ins = (torch.randn(512, 1024, generator=g),)
        target = torch.randint(0, 1024, (512,), generator=g)
    elif workload == "tel":
        m = nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.0, batch_first=True)
        ins = (torch.randn(32, 256, 512, generator=g),)
        target = torch.randn(32, 256, 512, generator=g)
    elif workload == "llama":
        m = _llama(dim=512, depth=4, heads=8, vocab=4096, seq=128)
        ins = (torch.randint(0, 4096, (4, 128), generator=g),)
        target = torch.randint(0, 4096, (4 * 128,), generator=g)
    else:
        import torchvision
        m = torchvision.models.resnet18()
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.1, generator=g)
                    mod.running_var.uniform_(0.5, 1.5, generator=g)
                    mod.weight.uniform_(0.5, 1.5, generator=g)
                    mod.bias.normal_(0, 0.1, generator=g)
        ins = (torch.randn(32, 3, 224, 224, generator=g),)
        target = torch.randint(0, 1000, (32,), generator=g)
    return m.eval(), ins, target


def one(cfg, time_it):
    import copy
    import torch
    import torch.nn.functional as F
    from truestar import ir, train as T
    from truestar.compiler import compile_program
    w, o, gmode = cfg.split(":")
    m, ins, target = build(w)
    row = {"config": cfg}
    m32 = copy.deepcopy(m).to(torch.float32).eval()
    prog = ir.import_module(m32, ins, name=f"{w}_train")
    loss = T.MSE() if w == "tel" else T.CrossEntropy()
    opt = T.SGD(SGD_LR, momentum=SGD_MOM) if o == "sgd" else T.AdamW(LR, betas=BETAS, eps=EPS, weight_decay=WD)
    st = T.build_train_step(prog, loss, opt, params=m32)
    names = [v.name for v in st.program.inputs]
    tgt = target.cuda() if w != "tel" else target.to("cuda", torch.float32)
    lst = []
    for i, n in enumerate(names):
        if i == 0:
            lst.append(ins[0].cuda())
        elif i == 1:
            lst.append(tgt)
        elif n == "lr":
            lst.append(torch.tensor(SGD_LR if o == "sgd" else LR, dtype=torch.float32, device="cuda"))
        else:
            t = st.initial[n]
            lst.append(t.to("cuda", torch.float32) if t.is_floating_point() else t.cuda())
    n_out = len(st.program.outputs)
    upd = {3 + k: 3 + k for k in range(n_out - 3)}
    try:
        cp = compile_program(st.program, "cuda", "float32", **({"graphs": "off"} if gmode == "off" else {}))
        call = (lambda gp: (lambda: gp(lst)))(cp.graphed(update=upd)) if gmode == "graph" else (lambda: cp(lst, update=upd))
        import time
        spent = []
        dv = cp._derived_verdict

        def timed_dv(*a, **kw):
            t1 = time.perf_counter()
            try:
                return dv(*a, **kw)
            finally:
                spent.append(time.perf_counter() - t1)
        cp._derived_verdict = timed_dv
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        row["first_call_s"] = time.perf_counter() - t0      # the verified call, the derivation included
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        row["passed"] = True
        row["derived_accepts"] = len(cp.derived_accepts)
        row["derivation_s"] = sum(spent)                  # the end to end derivation alone
        row["auto_graph_replays"] = cp.auto_graph_replays
    except Exception as e:  # noqa: BLE001  the refusal is the result
        lines = [ln.strip() for ln in f"{type(e).__name__}: {e}".splitlines() if ln.strip()]
        row["passed"] = False
        row["refusal"] = " | ".join(lines[:6])[:900]
        return row
    if not time_it:
        return row

    def timed(fn, n=20, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return sorted(ts)[n // 2]
    row["truestar_ms"] = timed(call)
    em = copy.deepcopy(m).to(torch.float32).cuda().eval()
    topt = (torch.optim.SGD(em.parameters(), lr=SGD_LR, momentum=SGD_MOM, fused=True) if o == "sgd"
            else torch.optim.AdamW(em.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fused=True))
    xe = ins[0].cuda()

    def estep():
        topt.zero_grad(set_to_none=True)
        out = em(xe)
        (F.mse_loss(out, tgt) if w == "tel" else F.cross_entropy(out, tgt)).backward()
        topt.step()
    row["eager_ms"] = timed(estep)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--json")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:
        print("ROW " + json.dumps(one(a.one, a.time)), flush=True)
        return
    out = open(a.json, "w", encoding="utf-8") if a.json else None
    for cfg in a.configs.split(","):
        cmd = [sys.executable, os.path.abspath(__file__), "--one", cfg] + (["--time"] if a.time else [])
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        rows = [l[4:] for l in p.stdout.splitlines() if l.startswith("ROW ")]
        r = json.loads(rows[0]) if rows else {"config": cfg, "passed": False,
                                               "refusal": "crashed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:300]}
        if out:
            out.write(json.dumps(r) + "\n")
            out.flush()
        if r["passed"]:
            t = f"  truestar {r['truestar_ms']:.2f} ms  eager {r['eager_ms']:.2f} ms" if "truestar_ms" in r else ""
            t += f"  first call {r.get('first_call_s', 0):.1f} s (derivation {r.get('derivation_s', 0):.1f} s)"
            print(f"{cfg:22} PASS{t}", flush=True)
        else:
            print(f"{cfg:22} REFUSED  {r['refusal']}", flush=True)


if __name__ == "__main__":
    main()
