"""Speculative decoding against plain greedy decoding: tokens bitwise equal, and tokens per second.

    PYTHONPATH=src python scripts/bench_specdec.py --pair DIR [--new 128] [--ks 2,3,4,6] [--json out.json]

The pair comes from scripts/specdec_pair.py. Both models are imported (examples/llama_paged_decode.py
dense step) and compiled in deterministic float32. Before timing, ``specdec.check_rows`` must report
the target row invariant (TS706) at every verify width. Then for each held out prompt (batch 1):

* baseline: ``specdec.greedy`` with the compiled target, one call per token;
* speculative: ``specdec.speculative`` with the compiled draft and target, ``k`` proposals per round.

Both run replayed from CUDA graphs when the build records them for the dense step (``cp.graphed``,
one graph per row count) and as ordinary compiled calls otherwise; the mode used is printed. The
speculative tokens must equal the greedy tokens on every prompt or the script stops (exit 1).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))


def load(path):
    import torch
    import llama_paged_decode as P
    d = torch.load(path, weights_only=True)
    m = P.Llama(P.Cfg(**d["cfg"]))
    m.load_state_dict(d["state"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--ks", default="2,3,4,6")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--graphs", choices=("auto", "off"), default="auto")
    ap.add_argument("--json")
    a = ap.parse_args()
    import torch
    import llama_paged_decode as P
    from truestar import specdec as SD
    from truestar.compiler import compile_program

    target, draft = load(os.path.join(a.pair, "target.pt")), load(os.path.join(a.pair, "draft.pt"))
    prompts = torch.load(os.path.join(a.pair, "prompts.pt"), weights_only=True)[: a.prompts]
    ks = [int(k) for k in a.ks.split(",")]
    t0 = prompts.shape[1]
    cap = -(-(t0 + a.new + max(ks) + 2) // 16) * 16
    progs = {}
    for name, m in (("target", target), ("draft", draft)):
        ir, _ = P.import_steps(m, capacity=cap)      # traced at batch 2 so B stays symbolic
        progs[name] = compile_program(ir, "cuda", "float32", deterministic=True)

    def cache(m):
        c = m.c
        k = torch.zeros(c.layers, 1, c.kv_heads, cap, c.head_dim, device="cuda")
        return k, torch.zeros_like(k)

    caches = {"target": cache(target), "draft": cache(draft)}
    mode = "ordinary compiled calls"
    steps = {n: (lambda cp: (lambda ids, pos, kc, vc: cp([ids, pos, kc, vc])))(progs[n]) for n in progs}
    if a.graphs == "auto":
        try:
            gsteps = {}
            for n in progs:
                gp = progs[n].graphed(update={2: 1, 3: 2})
                kc, vc = caches[n]
                gsteps[n] = (lambda gp, kc, vc: (lambda ids, pos, _k, _v: gp([ids, pos, kc, vc])))(gp, kc, vc)
                gsteps[n](prompts[:1].cuda(), torch.arange(t0, device="cuda"), kc, vc)
                for k in [1] + [k + 1 for k in ks]:
                    gsteps[n](prompts[:1, :k].cuda(), torch.arange(t0, t0 + k, device="cuda"), kc, vc)
            steps, mode = gsteps, "CUDA graph replay (one graph per row count)"
        except Exception as e:  # noqa: BLE001  the build refuses graphs for this step: say so, run ordinary
            mode = f"ordinary compiled calls (graphs refused: {str(e).splitlines()[0][:90]})"
    print(f"mode: {mode}")
    print(f"target {target.c.layers} layers dim {target.c.dim}; draft {draft.c.layers} layers dim {draft.c.dim}; "
          f"prompt {t0} bytes, {a.new} new tokens, {len(prompts)} prompts")

    def zero():
        for kc, vc in caches.values():
            kc.zero_()
            vc.zero_()

    rows = {}
    for k in ks:
        zero()
        r = SD.check_rows(steps["target"], prompts[:1], *caches["target"], rows=k + 1)
        rows[k] = r.to_dict()
        print(f"row invariance at {k + 1} rows: {r.codes[0]} (max diff {r.max_abs:.1e})")
        if not r.invariant:
            print("the target is not row invariant: speculative decoding would change its tokens; stopping")
            raise SystemExit(1)

    def run_greedy(p):
        zero()
        return SD.greedy(steps["target"], p, a.new, *caches["target"])

    def run_spec(p, k):
        zero()
        return SD.speculative(steps["target"], steps["draft"], p, a.new, k, *caches["target"], *caches["draft"])

    def timed(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t, out

    ref = [run_greedy(p[None]) for p in prompts]
    res = {"mode": mode, "row_checks": rows, "new_tokens": a.new, "prompts": len(prompts), "runs": []}
    base = []
    for _ in range(a.rounds):
        base.append(sum(timed(lambda: run_greedy(p[None]))[0] for p in prompts))
    tb = min(base)
    ntok = a.new * len(prompts)
    print(f"| method | tokens per s | acceptance | tokens per target call | bitwise equal to greedy |")
    print(f"|---|---|---|---|---|")
    print(f"| greedy (baseline) | {ntok / tb:.0f} | | 1.00 | |")
    res["runs"].append({"method": "greedy", "tokens_per_s": ntok / tb})
    for k in ks:
        outs = [run_spec(p[None], k) for p in prompts]
        equal = all(torch.equal(o[0], r) for o, r in zip(outs, ref))
        if not equal:
            print(f"k {k}: speculative tokens differ from greedy; stopping")
            raise SystemExit(1)
        acc = sum(o[1]["accepted"] for o in outs) / sum(o[1]["proposed"] for o in outs)
        tpr = sum(a.new - 1 for _ in outs) / sum(o[1]["rounds"] for o in outs)
        ts = min(sum(timed(lambda: run_spec(p[None], k))[0] for p in prompts) for _ in range(a.rounds))
        print(f"| speculative k={k} | {ntok / ts:.0f} ({tb / ts:.2f}x) | {acc:.2f} | {tpr:.2f} | yes, {len(prompts)} of {len(prompts)} |")
        res["runs"].append({"method": f"speculative k={k}", "k": k, "tokens_per_s": ntok / ts, "speedup": tb / ts,
                            "acceptance": acc, "tokens_per_round": tpr, "bitwise_equal": equal})
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
