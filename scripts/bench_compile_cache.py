"""Cold, warm in process and warm new process compile times with the persistent compile cache.

    python scripts/bench_compile_cache.py --device cuda            # every workload, the matrix below
    python scripts/bench_compile_cache.py --one mlp --cache D --out J [--device cpu]   # one process

For each workload the orchestrator runs, in fresh processes one at a time:

1. ``cold``: an empty TRUESTAR_CACHE_DIR (Triton's binaries live under it too); compile, first call,
   then a second compile of the same program in the same process ("warm in process") and its call.
2. ``warm``: the same cache directory in a new process ("warm new process").

Times, from the model in memory: ``export`` (ir.import_module, torch.export), ``compile``
(compile_program), ``first_call`` (the first call, verification included) and ``first_result``
(compile plus first call: the compiler's share, after the front end). Every output is written to
disk and compared bit for bit across the three runs. Models are small versions of the fair
benchmark's (<research>/FAIR_BENCH_2026_11) so the shared GPU is busy only briefly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

T_PROC = time.perf_counter()
HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = ("mlp", "encoder", "resnet18", "decode")


def build(workload):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(7)
    torch.manual_seed(0)
    if workload == "mlp":
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b, self.c = nn.Linear(1024, 2048), nn.Linear(2048, 2048), nn.Linear(2048, 1000)

            def forward(self, x):
                return self.c(F.gelu(self.b(F.gelu(self.a(x), approximate="tanh")), approximate="tanh"))
        return MLP(), (torch.randn(64, 1024, generator=g),), None
    if workload == "encoder":
        m = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True)
        return m, (torch.randn(8, 128, 256, generator=g),), None
    if workload == "resnet18":
        import torchvision
        m = torchvision.models.resnet18()
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.running_mean.normal_(0, 0.1, generator=g)
                    mod.running_var.uniform_(0.5, 1.5, generator=g)
        return m, (torch.randn(2, 3, 224, 224, generator=g),), None
    if workload == "decode":
        d, h, kv, hd, f, n = 512, 8, 2, 64, 1408, 256

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(1.0 + 0.1 * torch.randn(d, generator=g))

            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.w

        class Block(nn.Module):
            """One Llama style decode step: RMS norm, rotary q and k, grouped KV cache written at pos."""

            def __init__(self):
                super().__init__()
                self.n1, self.n2 = Norm(), Norm()
                self.q, self.k = nn.Linear(d, h * hd, bias=False), nn.Linear(d, kv * hd, bias=False)
                self.v, self.o = nn.Linear(d, kv * hd, bias=False), nn.Linear(h * hd, d, bias=False)
                self.up, self.gate = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False)
                self.down = nn.Linear(f, d, bias=False)

            @staticmethod
            def rot(x, cos, sin):
                return x * cos + torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), dim=-1) * sin

            def forward(self, x, cos, sin, pos, kc, vc):
                B = x.shape[0]
                a = self.n1(x)
                q = self.rot(self.q(a).view(B, 1, h, hd).transpose(1, 2), cos, sin)
                k = self.rot(self.k(a).view(B, 1, kv, hd).transpose(1, 2), cos, sin)
                v = self.v(a).view(B, 1, kv, hd).transpose(1, 2)
                kc = kc.index_copy(2, pos, k)
                vc = vc.index_copy(2, pos, v)
                o = F.scaled_dot_product_attention(q, kc, vc, enable_gqa=True)
                x = x + self.o(o.transpose(1, 2).reshape(B, 1, h * hd))
                b = self.n2(x)
                return x + self.down(F.silu(self.gate(b)) * self.up(b)), kc, vc
        inv = 1.0 / 10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
        fr = torch.tensor([float(n - 1)])[:, None] * inv[None]
        fr = torch.cat((fr, fr), dim=-1)
        ins = (torch.randn(2, 1, d, generator=g), fr.cos(), fr.sin(), torch.tensor([n - 1]),
               torch.randn(2, kv, n, hd, generator=g), torch.randn(2, kv, n, hd, generator=g))
        return Block(), ins, {4: 1, 5: 2}
    raise ValueError(workload)


def one(a):
    import torch
    t_torch = time.perf_counter() - T_PROC
    t0 = time.perf_counter()
    import truestar  # noqa: F401
    from truestar import ir
    from truestar.compiler import compile_program
    from truestar.kernel import tune as ktune
    try:
        from truestar import compile_cache as CC
    except ImportError:
        CC = None
    res = {"workload": a.one, "device": a.device, "import_torch_s": t_torch,
           "import_truestar_s": time.perf_counter() - t0}
    m, ins, upd = build(a.one)
    m = m.eval()
    t0 = time.perf_counter()
    if a.load:
        from truestar.ir import serialize
        prog = serialize.load(a.load).program
    else:
        prog = ir.import_module(m, ins, name=a.one)
    res["export_s"] = time.perf_counter() - t0
    if a.save:
        from truestar.ir import serialize
        serialize.save(prog, a.save)
    dev = torch.device(a.device)

    def run_once(tag):
        xs = [t.to(dev) for t in ins]
        sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
        sync()
        t1 = time.perf_counter()
        cp = compile_program(prog, a.device, "float32")
        t2 = time.perf_counter()
        outs = cp(xs, update=upd)
        sync()
        t3 = time.perf_counter()
        res[f"{tag}_compile_s"] = t2 - t1
        res[f"{tag}_first_call_s"] = t3 - t2
        res[f"{tag}_first_result_s"] = t3 - t1
        res[f"{tag}_verify_s"] = cp.verify_seconds + cp.kernel_verify_seconds
        res[f"{tag}_cache"] = dict(getattr(cp, "compile_cache", {}) or {})
        h = hashlib.sha256()
        for o in outs:
            h.update(o.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        res[f"{tag}_out_sha256"] = h.hexdigest()
        return cp

    run_once("first")
    res["process_to_first_result_s"] = time.perf_counter() - T_PROC
    res["tune_s"] = ktune.stats.seconds
    if a.inproc:
        run_once("again")
    res["cache_info"] = CC.info() if CC else None
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def orchestrate(a):
    rows = []
    py = sys.executable
    for w in a.workloads:
        d = tempfile.mkdtemp(prefix=f"tscc_{w}_")
        env = dict(os.environ, TRUESTAR_CACHE_DIR=d)
        env.pop("TRITON_CACHE_DIR", None)
        if a.triton_in_cache:    # for a Truestar without the compile cache: Triton's binaries start cold too
            env["TRITON_CACHE_DIR"] = os.path.join(d, "triton", "bin")
        env.update({k: v for k, v in (x.split("=", 1) for x in a.env)})
        row = {"workload": w}
        tsx = os.path.join(d, "_program.tsx")
        for phase in ("cold", "warm", "warm_tsx"):
            out = os.path.join(d, f"_{phase}.json")
            cmd = [py, os.path.abspath(__file__), "--one", w, "--device", a.device, "--out", out]
            if phase == "cold":
                cmd += ["--inproc", "--save", tsx]
            if phase == "warm_tsx":
                cmd += ["--load", tsx]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if p.returncode:
                print(p.stdout[-3000:], p.stderr[-6000:])
                raise SystemExit(f"{w} {phase} failed")
            r = json.load(open(out))
            r["wall_s"] = time.perf_counter() - t0
            row[phase] = r
        c, wm, wt = row["cold"], row["warm"], row["warm_tsx"]
        row["bitwise_equal"] = (c["first_out_sha256"] == c["again_out_sha256"] == wm["first_out_sha256"]
                                == wt["first_out_sha256"])
        rows.append(row)
        print(f"{w:9s} export {c['export_s']:.2f}/{wm['export_s']:.2f}  cold {c['first_first_result_s']:.2f} "
              f"(verify {c['first_verify_s']:.2f})  warm in process {c['again_first_result_s']:.3f} "
              f"(verify {c['again_verify_s']:.3f})  warm new process {wm['first_first_result_s']:.3f} "
              f"(verify {wm['first_verify_s']:.3f}, import {wm['import_truestar_s']:.2f})  "
              f"| from .tsx: load {wt['export_s']:.3f} first result {wt['first_first_result_s']:.3f} process start "
              f"to first result {wt['process_to_first_result_s']:.2f} (torch import {wt['import_torch_s']:.2f})  "
              f"bitwise {row['bitwise_equal']}  verify {wm['first_cache'].get('verify')}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out")
    ap.add_argument("--inproc", action="store_true")
    ap.add_argument("--save", help="also write the imported program to this .tsx")
    ap.add_argument("--load", help="read the program from this .tsx instead of torch.export")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE for the child processes")
    ap.add_argument("--json")
    ap.add_argument("--triton-in-cache", action="store_true",
                    help="set TRITON_CACHE_DIR inside each fresh cache directory (measuring the code before the cache)")
    a = ap.parse_args()
    if a.one:
        one(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
