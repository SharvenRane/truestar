"""MoE measurements for docs/MOE.md: determinism, CPU vs CUDA, float64 error, grouped experts speed.

    PYTHONPATH=src python scripts/moe_bench.py [--json out.json]

Needs CUDA for the device parts (they are skipped and said so without it) and transformers 5.x for
the family parts. TF32 is off for every float32 matmul here; timings are the median of 30 CUDA
event timed calls after 5 warm up calls, with the float64 check off (TRUESTAR_VERIFY=0) so the
grouped kernel is timed alone, and the inputs and weights are drawn once per config.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics

import torch
import torch.nn as nn

from truestar import ir
from truestar.moe import experts as X
from truestar.moe import grouped_experts, reference_experts, restore, use_grouped
from truestar.moe.ties import router_ties

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CUDA = torch.cuda.is_available()


def _args(T, E, H, inter, k, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g)
    logits = torch.randn(T, E, generator=g)
    w, idx = torch.softmax(logits, -1).topk(k, -1)
    w = w / w.sum(-1, keepdim=True)
    gu = torch.randn(E, 2 * inter, H, generator=g) / H ** 0.5
    dn = torch.randn(E, H, inter, generator=g) / inter ** 0.5
    return [t.to(device, dtype) if t.is_floating_point() else t.to(device) for t in (x, idx, w, gu, dn)]


def _hf_eager(x, idx, w, gu, dn, act):
    """transformers 5.24's eager experts loop (one_hot mask, where per hit expert, index_add)."""
    out = torch.zeros_like(x)
    mask = torch.nn.functional.one_hot(idx, num_classes=gu.shape[0]).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit:
        e = e[0]
        pos, tok = torch.where(mask[e])
        gate, up = torch.nn.functional.linear(x[tok], gu[e]).chunk(2, dim=-1)
        y = torch.nn.functional.linear(act(gate) * up, dn[e]) * w[tok, pos, None]
        out.index_add_(0, tok, y.to(out.dtype))
    return out


def _time(fn, n=30, warm=5):
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
    return statistics.median(ts)


def kernel_section(res):
    act = nn.SiLU()
    # float64 error of the grouped experts in float32, CPU and CUDA, and determinism
    for dev in ["cpu"] + (["cuda"] if CUDA else []):
        x, idx, w, gu, dn = _args(256, 8, 128, 256, 2, torch.float32, dev)
        ref = reference_experts(x, idx, w, gu, dn, act)
        X.reset()
        out = grouped_experts(x, idx, w, gu, dn, act)
        scale = ref.abs().max().item()
        res[f"grouped_fp32_vs_fp64_{dev}"] = (out.double().cpu() - ref).abs().max().item() / scale
        x64, _, w64, gu64, dn64 = _args(256, 8, 128, 256, 2, torch.float64, dev)
        res[f"grouped_fp64_vs_fp64_{dev}"] = (grouped_experts(x64, idx, w64, gu64, dn64, act).cpu()
                                              - reference_experts(x64, idx, w64, gu64, dn64, act)).abs().max().item() / scale

        def run():
            leaves = [t.clone().requires_grad_(True) for t in (x, w, gu, dn)]
            o = grouped_experts(leaves[0], idx, leaves[1], leaves[2], leaves[3], act)
            o.backward(torch.ones_like(o))
            return [o.detach()] + [t.grad for t in leaves]

        a, b = run(), run()
        res[f"grouped_two_runs_bitwise_{dev}"] = all(torch.equal(p, q) for p, q in zip(a, b))
        if dev == "cuda":
            cpu = [t.cpu() for t in a]
            res["grouped_cpu_vs_cuda_bitwise"] = torch.equal(cpu[0], grouped_experts(
                x.cpu(), idx.cpu(), w.cpu(), gu.cpu(), dn.cpu(), act))
            res["grouped_cpu_vs_cuda_max_rel"] = (cpu[0] - grouped_experts(
                x.cpu(), idx.cpu(), w.cpu(), gu.cpu(), dn.cpu(), act)).abs().max().item() / scale


def _hf_experts(E, H, inter, k, impl, gu, dn):
    from transformers import AutoConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    cfg = AutoConfig.for_model("qwen3_moe", hidden_size=H, moe_intermediate_size=inter, num_experts=E,
                               num_experts_per_tok=k, experts_implementation=impl)
    m = Qwen3MoeExperts(cfg).to(gu.device, gu.dtype)
    m.gate_up_proj.data, m.down_proj.data = gu, dn
    return m


def speed_section(res):
    if not CUDA:
        res["speed"] = "skipped: no CUDA"
        return
    os.environ["TRUESTAR_VERIFY"] = "0"
    act = nn.SiLU()
    configs = {"small": (512, 8, 256, 512, 2), "medium": (8192, 32, 1024, 1408, 4)}
    out = {}
    for name, (T, E, H, inter, k) in configs.items():
        for dtype in (torch.float32, torch.bfloat16):
            x, idx, w, gu, dn = _args(T, E, H, inter, k, dtype, "cuda")
            row = {"T": T, "E": E, "H": H, "I": inter, "k": k}
            row["grouped_ms"] = _time(lambda: grouped_experts(x, idx, w, gu, dn, act))
            row["hf_eager_ms"] = _time(lambda: _hf_eager(x, idx, w, gu, dn, act))
            for impl in ("eager", "grouped_mm"):     # transformers' own experts module, both modes
                try:
                    hf = _hf_experts(E, H, inter, k, impl, gu, dn)
                    row[f"hf_{impl}_module_ms"] = _time(lambda: hf(x, idx, w))
                except Exception as e:  # noqa: BLE001
                    row[f"hf_{impl}_module_ms"] = f"not run: {type(e).__name__}: {str(e)[:120]}"
            same = torch.equal(grouped_experts(x, idx, w, gu, dn, act), grouped_experts(x, idx, w, gu, dn, act))
            row["grouped_bitwise_repeat"] = same
            out[f"{name}_{str(dtype).replace('torch.', '')}"] = row
    os.environ.pop("TRUESTAR_VERIFY")
    res["speed"] = out


def family_section(res):
    try:
        from truestar import testing_hf
    except ImportError:
        res["families"] = "skipped: no transformers"
        return
    fams = testing_hf.available(getattr(testing_hf, "MOE_2026", {}))
    out = {}
    for name in fams:
        model, args = testing_hf.build(name)
        row = {}
        m64 = copy.deepcopy(model).double()
        a64 = tuple(a.double() if a.is_floating_point() else a for a in args)
        p64 = ir.import_module(m64, a64)
        with torch.no_grad(), ir.reference.exact_float32_casts():
            want = m64(*a64)[0]
        row["ir_fp64_rel"] = ((ir.run(p64, *a64)[0] - want).abs().max() / want.abs().max()).item()
        prog = ir.import_module(model, args)
        from truestar.compiler import compile_program
        outs = {}
        for dev in ["cpu"] + (["cuda"] if CUDA else []):
            cp = compile_program(prog, dev, "float32")
            dargs = [a.to(dev) for a in args]
            o1 = cp(dargs)[0].cpu()
            o2 = cp(dargs)[0].cpu()
            outs[dev] = o1
            row[f"compiled_{dev}_two_runs_bitwise"] = torch.equal(o1, o2)
            row[f"compiled_{dev}_fp64_rel"] = ((o1.double() - want).abs().max() / want.abs().max()).item()
        if CUDA:
            row["compiled_cpu_vs_cuda_bitwise"] = torch.equal(outs["cpu"], outs["cuda"])
            # eager with the grouped experts and the tie rule, on each device: same experts chosen?
            picks = {}
            for dev in ("cpu", "cuda"):
                m = copy.deepcopy(model).to(dev)
                use_grouped(m)
                seen = []
                hooks = [mod.register_forward_hook(lambda _m, _a, o: seen.append(o[2].cpu()))
                         for p, mod in m.named_modules() if p.endswith(".gate") and hasattr(mod, "top_k")]
                with torch.no_grad():
                    o1 = m(*[a.to(dev) for a in args])[0].cpu()
                    o2 = m(*[a.to(dev) for a in args])[0].cpu()
                for h in hooks:
                    h.remove()
                restore(m)
                picks[dev] = seen[: len(hooks)]
                row[f"eager_grouped_{dev}_two_runs_bitwise"] = torch.equal(o1, o2)
            row["routing_cpu_vs_cuda_same_experts"] = all(
                torch.equal(torch.sort(a, -1).values, torch.sort(b, -1).values) for a, b in zip(picks["cpu"], picks["cuda"]))
        for dt in (torch.float32, torch.bfloat16):
            rep = router_ties(model, args, dtype=dt, device="cuda" if CUDA else "cpu")
            row[f"near_ties_{str(dt).replace('torch.', '')}"] = sum(len(layer.ties) for layer in rep.layers)
            row[f"tokens_{str(dt).replace('torch.', '')}"] = sum(layer.tokens for layer in rep.layers)
        out[name] = row
    res["families"] = out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="")
    p.add_argument("--only", default="", help="kernel, families or speed")
    a = p.parse_args()
    res = {"torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if CUDA else None}
    for name, fn in (("kernel", kernel_section), ("families", family_section), ("speed", speed_section)):
        if not a.only or a.only == name:
            fn(res)
    text = json.dumps(res, indent=1, default=str)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
