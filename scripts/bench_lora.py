"""LoRA and QLoRA training step benchmark (docs/LORA.md).

    python scripts/bench_lora.py --device cpu  [--dim 128 --depth 2 --steps 30]
    python scripts/bench_lora.py --device cuda --dim 512 --depth 4

A Llama style model with random weights from a config (nothing is downloaded) learns a synthetic
task: next token = a fixed random permutation of the current one. Measured per configuration: the
loss at the first and last step, the median step time after warm up, peak device memory (CUDA) and
the carried bytes (trained tensors, gradients, AdamW slots, frozen base). Configurations:

* truestar LoRA (q_proj, v_proj; rank 8, alpha 16, dropout 0.05), compiled executor;
* truestar QLoRA, the same adapters on an int4:g64 base (every Linear but the head);
* truestar full fine tuning of the same model, compiled executor;
* eager PyTorch full fine tuning of the same model and weights (torch.optim.AdamW);
* a hand written LoRA in eager PyTorch (frozen base, the same adapters and initial values).

Then the truestar LoRA run is repeated from the same seed and its adapters compared bit for bit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import truestar  # noqa: E402
from truestar import Tensor, lora, nn, ops, optim  # noqa: E402
from truestar.nn.llama import Llama  # noqa: E402


# ----------------------------------------------------------------------------- the eager PyTorch baseline


class TorchLlama(torch.nn.Module):
    """The same decoder in plain PyTorch, weights copied from a truestar tree."""

    def __init__(self, net: Llama, p, lora_paths=(), lm=None, a=None):
        super().__init__()
        self.net = net
        self.w = torch.nn.ParameterDict()
        self.lora = torch.nn.ParameterDict()
        self.scale = {}
        for path, t in lora._leaves(p):
            self.w[path.replace(".", "__")] = torch.nn.Parameter(t.clone())
        for path in lora_paths:
            node = lora._get(a, path)
            self.lora[path.replace(".", "__") + "__a"] = torch.nn.Parameter(node["lora_a"].clone())
            self.lora[path.replace(".", "__") + "__b"] = torch.nn.Parameter(node["lora_b"].clone())
            self.scale[path] = lm.adapted[path].scale
        hd = net.dim // net.heads
        k = torch.arange(0, hd, 2, dtype=torch.float64)
        self.register_buffer("inv", torch.exp(k * (-math.log(net.base) / hd)), persistent=False)

    def W(self, path):
        return self.w[path.replace(".", "__")]

    def lin(self, path, x):
        y = x @ self.W(path + ".w").T
        key = path.replace(".", "__")
        if key + "__a" in self.lora:
            h = F.dropout(x, 0.05, self.training)
            y = y + (h @ self.lora[key + "__a"].T @ self.lora[key + "__b"].T) * self.scale[path]
        return y

    def rms(self, path, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.net.eps) * self.W(path + ".w")

    def forward(self, ids):
        n = self.net
        B, T = ids.shape
        hd = n.dim // n.heads
        x = self.W("embed_tokens.w")[ids]
        ang = torch.arange(T, device=ids.device, dtype=torch.float64)[:, None] * self.inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
        sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)

        def rope(t):
            return t * cos + torch.cat([-t[..., hd // 2:], t[..., :hd // 2]], -1) * sin
        for i in range(n.depth):
            b = f"l{i}."
            h = self.rms(b + "input_layernorm", x)
            q = self.lin(b + "self_attn.q_proj", h).view(B, T, n.heads, hd).transpose(1, 2)
            k = self.lin(b + "self_attn.k_proj", h).view(B, T, n.kv_heads, hd).transpose(1, 2)
            v = self.lin(b + "self_attn.v_proj", h).view(B, T, n.kv_heads, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True, enable_gqa=True)
            x = x + self.lin(b + "self_attn.o_proj", a.transpose(1, 2).reshape(B, T, n.dim))
            h = self.rms(b + "post_attention_layernorm", x)
            x = x + self.lin(b + "mlp.down_proj", F.silu(self.lin(b + "mlp.gate_proj", h)) * self.lin(b + "mlp.up_proj", h))
        return self.lin("lm_head", self.rms("norm", x))


# ----------------------------------------------------------------------------- helpers


def _executor(dev):
    """The compiled executor. AdamW's first update is ill conditioned in float32 (m / (sqrt(v) + eps) on
    gradients near eps): outputs within 4x of an independent float32 evaluation's distance from float64
    are accepted and reported (TS782, docs/COMPILER.md), every other output keeps its float64 bar."""
    from truestar.compiler import CompiledExecutor
    return CompiledExecutor(dev, accept_ill_conditioned=True)


def _conditioning(step):
    c = getattr(step._run, "conditioning", None)
    return len(c) if c else 0


def _to(tree, dev):
    return {k: (_to(v, dev) if isinstance(v, dict) else v.to(dev)) for k, v in tree.items()}


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _peak_reset(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return 0


def _peak(dev, start):
    return int(torch.cuda.max_memory_allocated() - start) if dev == "cuda" else None


def batches(vocab, B, T, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(vocab, generator=g)
    out = []
    for _ in range(steps):
        x0 = torch.randint(0, vocab, (B, 1), generator=g)
        seq = [x0]
        for _ in range(T):
            seq.append(perm[seq[-1]])
        s = torch.cat(seq, 1)
        out.append((s[:, :T].contiguous(), s[:, 1:].contiguous()))
    return out


STEADY = {}


def timed(fn, data, warm, dev="cpu"):
    """Losses, median step ms after `warm` steps, first call seconds; STEADY["peak"] is the peak device
    memory over steps 2 on (after the first call's compile and float64 check)."""
    times, losses = [], []
    for i, (x, y) in enumerate(data):
        t0 = time.perf_counter()
        losses.append(fn(x, y))
        times.append(time.perf_counter() - t0)
        if i == 0:
            STEADY["first_abs"] = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
            STEADY["start"] = _peak_reset(dev)
    STEADY["peak"] = _peak(dev, STEADY["start"])
    return losses, statistics.median(times[warm:]) * 1e3, times[0]


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    dev = args.device
    V = args.vocab
    net = Llama(V, args.dim, args.depth, args.heads, args.kv, args.ffn)
    p = net.init(seed=0)
    data = batches(V, args.batch, args.seq, args.steps)
    ddata = [(x.to(dev), y.to(dev)) for x, y in data]
    res = {"device": dev, "torch": torch.__version__, "config": vars(args),
           "model_params": sum(t.numel() for _, t in lora._leaves(p))}
    if dev == "cuda":
        res["gpu"] = torch.cuda.get_device_name()
    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

    lm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=8, alpha=16.0, dropout=0.05)
    qlm = lora.wrap(net, targets=("q_proj", "v_proj"), rank=8, alpha=16.0, dropout=0.05, quant="int4:g64")
    a0 = lm.init(seed=1)

    def ts_lora(model, name):
        base = model.prepare_base(p)

        @truestar.native(seed=11)
        def loss(a: model.spec, p: model.base_spec, ids: Tensor["B", "T", "i64"], y: Tensor["B", "T", "i64"],
                 training: bool = False) -> Tensor[()]:
            logits = model(p, a, ids, training=training)
            return nn.cross_entropy(ops.reshape(logits, [-1, V]), ops.reshape(y, [-1]))
        start = _peak_reset(dev)
        t0 = time.perf_counter()
        step = truestar.train_step(loss, optim.AdamW(lr=args.lr, weight_decay=0.0), params="a",
                                   static={"training": True}, executor=_executor(dev))
        st = {"ts": step.init(_to(a0, dev))}
        bd = _to(base, dev)
        build = time.perf_counter() - t0

        def one(x, y):
            st["ts"], r = step(st["ts"], bd, x, y)
            return r.loss
        losses, ms, first = timed(one, ddata, args.warm, dev)
        mem = lora.memory_report(model, base, p)
        return {"loss_first": losses[0], "loss_last": losses[-1], "step_ms": ms, "first_call_s": first + build,
                "first_call_peak_bytes": (STEADY["first_abs"] - start) if dev == "cuda" else None, "steady_peak_bytes": STEADY["peak"], "carried": mem, "ill_conditioned_outputs": _conditioning(step)}, st["ts"]

    if want("truestar_lora"):
        res["truestar_lora"], ts1 = ts_lora(lm, "lora")
        # determinism: the same run again, adapters compared bit for bit
        again, ts2 = ts_lora(lm, "lora")
        try:
            lora.assert_bitwise(ts1.params, ts2.params)
            res["bitwise_two_runs"] = True
        except lora.LoRAError as e:
            res["bitwise_two_runs"] = str(e)
        res["truestar_lora"]["losses_equal_two_runs"] = again["loss_last"] == res["truestar_lora"]["loss_last"]
    if want("truestar_qlora"):
        res["truestar_qlora_int4"], _ = ts_lora(qlm, "qlora")

    if want("truestar_full"):
        @truestar.native(seed=11)
        def floss(p: net.spec, ids: Tensor["B", "T", "i64"], y: Tensor["B", "T", "i64"]) -> Tensor[()]:
            return nn.cross_entropy(ops.reshape(net(p, ids), [-1, V]), ops.reshape(y, [-1]))
        start = _peak_reset(dev)
        t0 = time.perf_counter()
        step = truestar.train_step(floss, optim.AdamW(lr=args.lr, weight_decay=0.0), executor=_executor(dev))
        st = {"ts": step.init(_to(p, dev))}
        build = time.perf_counter() - t0

        def one(x, y):
            st["ts"], r = step(st["ts"], x, y)
            return r.loss
        try:
            losses, ms, first = timed(one, ddata, args.warm, dev)
        except Exception as e:   # the compiled step's own end to end check can refuse it: record, go on
            res["truestar_full"] = {"error": str(e).splitlines()[0] + " " + " ".join(str(e).splitlines()[1:3])}
            losses = None
        if losses is not None:
            res["truestar_full"] = {"loss_first": losses[0], "loss_last": losses[-1], "step_ms": ms,
                                    "first_call_s": first + build, "first_call_peak_bytes": (STEADY["first_abs"] - start) if dev == "cuda" else None, "steady_peak_bytes": STEADY["peak"],
                                    "ill_conditioned_outputs": _conditioning(step)}

    def torch_run(paths):
        start = _peak_reset(dev)
        m = TorchLlama(net, p, paths, lm, a0).to(dev).train()
        if paths:
            for w in m.w.values():
                w.requires_grad_(False)
            params = list(m.lora.values())
        else:
            params = list(m.w.values())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

        def one(x, y):
            loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return float(loss.detach())
        losses, ms, first = timed(one, ddata, args.warm, dev)
        return {"loss_first": losses[0], "loss_last": losses[-1], "step_ms": ms, "first_call_s": first,
                "first_call_peak_bytes": (STEADY["first_abs"] - start) if dev == "cuda" else None, "steady_peak_bytes": STEADY["peak"]}

    if want("torch_full"):
        res["torch_full"] = torch_run(())
    if want("torch_lora"):
        res["torch_lora"] = torch_run(tuple(sorted(lm.adapted)))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
