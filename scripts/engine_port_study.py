"""Measurements for docs/PORT_ENGINES.md: llama.cpp and cross framework port checks.

Usage: PYTHONPATH=src;tests python scripts/engine_port_study.py OUT_DIR [--seeds N] [--skip-frameworks]

Engine part: clean ports (F32, F16 weights, Q8_0, Q4_K_M, each with an f32 and an f16 KV cache)
and injected faults (file: rope theta, norm eps, rope scaling, a transposed square weight, q or k
left unpermuted, a scaled weight, BOS; engine: one node perturbed by a relative 1e-4 to 1e-2 at a
random node) over several seeds. Baselines on the same runs:
  final: the final logits against the reference, with the largest clean error of the same file
         type as the threshold (what comparing outputs can do);
  naive: each engine node against the float64 reference model's own activation (accumulated
         error), threshold the largest clean error at that node for the same file type;
  no rounding model: the own error replay without the allowed activation rounding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch

import engine_models as em
from truestar.engines import engine_port as EP
from truestar.engines import llama_ref as R
from truestar.engines import llamacpp

IDS = em.reference_tokenize(em.PROMPT)
TYPES = ["F32", "F16", "Q8_0", "Q4_K_M"]


def make(model, d, name, ftype, **kw):
    base = em.to_gguf(model, os.path.join(d, f"{name}_{ftype}_src.gguf"), float_type="F16" if ftype == "F16" else "F32", **kw)
    if ftype in ("F32", "F16"):
        return base
    out = os.path.join(d, f"{name}_{ftype}.gguf")
    llamacpp.quantize(base, out, ftype)
    return out


def naive_errors(model, rep_path, mutate=None):
    """Accumulated error per step: engine node against the reference model's own activation."""
    g = EP.gguf.read(rep_path)
    cfg = R.LlamaConfig.from_hf(model.config)
    G = EP._gguf_weights(g)
    ref = EP._ref_params(model)
    _, _, w_ref, _ = EP.match_weights(G, ref, cfg, "llama")
    acts_ref = R.forward(cfg, w_ref, torch.tensor(IDS))
    run = llamacpp.run(rep_path, IDS, mutate=mutate)
    acts, _ = EP._engine_acts(run, cfg, IDS, "llama")
    return {k: EP._rel(acts[k], acts_ref[k]) for k in acts if k != "ids"}


def engine_part(d, seeds):
    rows, clean_final, clean_naive, runtimes = [], {}, {}, []
    nodes = [("ffn_up-{l}", "MUL_MAT", "{l}.up"), ("ffn_gate-{l}", "MUL_MAT", "{l}.gate"), ("Qcur-{l}", "ROPE", "{l}.q_rope"),
             ("Kcur-{l}", "ROPE", "{l}.k_rope"), ("kqv_out-{l}", "CONT", "{l}.attn"), ("attn_norm-{l}", "MUL", "{l}.attn_norm"),
             ("ffn_out-{l}", "MUL_MAT", "{l}.down"), ("Vcur-{l}", "MUL_MAT", "{l}.v")]
    for seed in range(seeds):
        model = em.build(seed=seed)
        rng = random.Random(seed)
        sd = os.path.join(d, f"s{seed}")
        os.makedirs(sd, exist_ok=True)
        clean = {t: make(model, sd, "clean", t) for t in TYPES}
        # clean ports
        for t in TYPES:
            for kv in ("f32", "f16"):
                t0 = time.perf_counter()
                r = EP.engine_port_check(model, clean[t], IDS, kv_type=kv)
                runtimes.append((time.perf_counter() - t0, r.seconds))
                bad = [c for c in r.codes if c in ("TS640", "TS641", "TS642", "TS643", "TS646", "TS649", "TS657")]
                worst = max(s.rel_err / s.threshold for s in r.engine_steps)
                rows.append({"seed": seed, "case": "clean", "type": t, "kv": kv, "status": r.status, "false_alarm": bool(bad),
                             "worst_ratio": worst, "final": r.decomposition["total"], "quant": r.decomposition["quantization"],
                             "top1": r.decisions_total.top1_agree / r.decisions_total.positions})
                if kv == "f32":
                    clean_final[t] = max(clean_final.get(t, 0.0), r.decomposition["total"])
                    ne = naive_errors(model, clean[t])
                    cn = clean_naive.setdefault(t, {})
                    for k, v in ne.items():
                        cn[k] = max(cn.get(k, 0.0), v)
        # file faults
        sq = [f"blk.{l}.{n}.weight" for l in range(em.LAYERS) for n in ("attn_q", "attn_output")]
        faults = [("theta", {"theta": rng.choice([5e5, 1e6, 2e4])}, "settings"),
                  ("eps", {"eps": rng.choice([1e-6, 1e-4])}, "settings"),
                  ("rope_scale", {"rope_scale": rng.choice([2.0, 4.0])}, "settings"),
                  ("transpose", {"transpose": [rng.choice(sq)]}, "weights"),
                  ("unpermuted", {"skip_permute": [f"blk.{rng.randrange(em.LAYERS)}.{rng.choice(['attn_q', 'attn_k'])}.weight"]}, "weights"),
                  ("scaled", {"scale_weight": {f"blk.{rng.randrange(em.LAYERS)}.ffn_up.weight": rng.choice([0.5, 2.0])}}, "weights"),
                  ("no_bos", {"add_bos": False}, "tokenizer")]
        for name, kw, blame in faults:
            for t in TYPES:
                p = make(model, sd, name, t, **kw)
                r = EP.engine_port_check(model, p, IDS, prompt=em.PROMPT if name == "no_bos" else None)
                fin = r.decomposition.get("total") if r.decomposition else None
                rows.append({"seed": seed, "case": name, "type": t, "status": r.status, "blame": r.blame,
                             "caught": r.status == "divergent", "right_blame": r.blame == blame,
                             "engine_false_blame": "TS640" in r.codes,
                             "final": fin, "final_baseline_caught": (fin or 0) > clean_final[t]})
        # engine faults
        for t in TYPES:
            for _ in range(3):
                pat, op, step = rng.choice(nodes)
                layer = rng.randrange(em.LAYERS)
                node, key = pat.format(l=layer), step.format(l=layer)
                frac = rng.choice([1e-4, 1e-3, 1e-2])

                def mutate(tap, a, node=node, op=op, frac=frac):
                    if tap.name == node and tap.op == op:
                        a = a.copy()
                        a.reshape(-1)[11] += frac * np.abs(a).max()
                        return a
                r = EP.engine_port_check(model, clean[t], IDS, mutate=mutate)
                fe = r.first_engine()
                ne = naive_errors(model, clean[t], mutate)
                naive_caught = ne[key] > clean_naive[t][key]
                rows.append({"seed": seed, "case": "engine_node", "type": t, "node": node, "size": frac,
                             "caught": r.status == "divergent", "right_blame": bool(fe and fe.key == key and r.blame == "engine"),
                             "final": r.decomposition["total"], "final_baseline_caught": r.decomposition["total"] > clean_final[t],
                             "naive_baseline_caught": naive_caught, "own_error": fe.rel_err if fe else None})
    return rows, runtimes


def no_rounding_baseline(d):
    model = em.build(seed=0)
    out = {}
    orig = EP._paths

    def f32_only(st, wt, kv):
        yield ("f32", None, None)
    for t in ("Q8_0", "Q4_K_M", "F16"):
        p = make(model, os.path.join(d, "s0"), "clean", t)
        EP._paths = f32_only
        try:
            r = EP.engine_port_check(model, p, IDS, kv_type="f16" if t == "F16" else "f32")
        finally:
            EP._paths = orig
        out[t] = {"false_alarm": "TS640" in r.codes, "steps_over": sum(s.over for s in r.engine_steps),
                  "worst_own": max(s.rel_err for s in r.engine_steps)}
    return out


def framework_part(seeds, tmp):
    from truestar.engines import xport

    tests = os.path.dirname(os.path.abspath(em.__file__))
    ids = torch.tensor([IDS])

    def cand(module, **kw):
        f = os.path.join(tmp, f"c_{module}_{abs(hash(repr(sorted(kw.items()))))}.py")
        args = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(f"import sys\nsys.path.insert(0, {tests!r})\nfrom {module} import build\n"
                     f"def get_model(w):\n    return build(w{', ' + args if args else ''})\n")
        return f
    rows = []
    cases = [("clean", "xport_flax", {}, None), ("eps", "xport_flax", {"eps": 1e-6}, "blocks.0.attn_norm"),
             ("theta", "xport_flax", {"theta": 5e5}, "blocks.0.attn"), ("pairs", "xport_flax", {"pairs": True}, "blocks.0.attn"),
             ("untransposed", "xport_flax", {"untransposed": ("model.layers.1.self_attn.q_proj.weight",)}, "blocks.1.attn.wq"),
             ("clean", "xport_keras", {}, None), ("eps", "xport_keras", {"eps": 1e-6}, "blocks.0.n1"),
             ("backend_bug", "xport_keras", {"backend_bug": "torch"}, "blocks.0.attn"),
             ("untransposed", "xport_keras", {"untransposed": ("model.layers.0.self_attn.o_proj.weight",)}, "blocks.0.attn.o")]
    for seed in range(seeds):
        model = em.build(seed=seed)
        for name, module, kw, expect in cases:
            kind = "jax" if module == "xport_flax" else "keras"
            t0 = time.perf_counter()
            try:
                r = xport.framework_port_check(model, ids, kind, cand(module, **kw), backends=["jax", "torch"] if kind == "keras" else None)
            except xport.FrameworkError as exc:
                rows.append({"seed": seed, "case": name, "module": module, "error": str(exc)[:200]})
                continue
            for b in r.backends:
                row = {"seed": seed, "case": name, "framework": b.label, "status": b.status,
                       "blamed": b.blamed.candidate if b.blamed else None, "final": b.output_rel_err,
                       "seconds": time.perf_counter() - t0, "codes": [i.code for i in b.issues]}
                if name == "clean":
                    row["false_alarm"] = b.status != "ok"
                elif name == "backend_bug" and b.label != "keras:torch":
                    row["false_alarm"] = b.status != "ok"
                else:
                    row["caught"] = b.status == "divergent"
                    row["right_module"] = b.blamed is not None and b.blamed.candidate == expect
                rows.append(row)
    return rows


def summarize(rows, runtimes, norm, frows):
    S = {}
    clean = [r for r in rows if r["case"] == "clean"]
    S["engine_clean_runs"] = len(clean)
    S["engine_false_alarms"] = sum(r["false_alarm"] for r in clean)
    S["engine_worst_clean_ratio"] = max(r["worst_ratio"] for r in clean)
    for t in TYPES:
        c = [r for r in clean if r["type"] == t and r["kv"] == "f32"]
        S[f"clean_{t}"] = {"final_rel_max": max(r["final"] for r in c), "top1_min": min(r["top1"] for r in c),
                           "quant_rel_max": max(r["quant"] for r in c)}
    faults = [r for r in rows if r["case"] not in ("clean", "engine_node")]
    S["file_faults"] = len(faults)
    S["file_faults_caught"] = sum(r["caught"] for r in faults)
    S["file_faults_right_blame"] = sum(r["right_blame"] for r in faults)
    S["file_faults_engine_false_blame"] = sum(r["engine_false_blame"] for r in faults)
    S["file_faults_final_baseline_caught"] = sum(r["final_baseline_caught"] for r in faults)
    eng = [r for r in rows if r["case"] == "engine_node"]
    S["engine_faults"] = len(eng)
    S["engine_faults_caught"] = sum(r["caught"] for r in eng)
    S["engine_faults_right_node"] = sum(r["right_blame"] for r in eng)
    S["engine_faults_final_baseline_caught"] = sum(r["final_baseline_caught"] for r in eng)
    S["engine_faults_naive_baseline_caught"] = sum(r["naive_baseline_caught"] for r in eng)
    for t in TYPES:
        e = [r for r in eng if r["type"] == t]
        S[f"engine_faults_{t}"] = {"n": len(e), "caught": sum(r["caught"] for r in e), "right": sum(r["right_blame"] for r in e),
                                   "naive": sum(r["naive_baseline_caught"] for r in e), "final": sum(r["final_baseline_caught"] for r in e)}
    tot = [x[0] for x in runtimes]
    S["engine_check_seconds_median"] = statistics.median(tot)
    S["engine_check_seconds_max"] = max(tot)
    S["engine_run_seconds_median"] = statistics.median(x[1].get("engine_tapped", 0) for x in runtimes)
    S["no_rounding_model"] = norm
    if frows:
        fc = [r for r in frows if "false_alarm" in r]
        ff = [r for r in frows if "caught" in r]
        S["framework_clean_runs"] = len(fc)
        S["framework_false_alarms"] = sum(r["false_alarm"] for r in fc)
        S["framework_faults"] = len(ff)
        S["framework_faults_caught"] = sum(r["caught"] for r in ff)
        S["framework_faults_right_module"] = sum(r["right_module"] for r in ff)
        S["framework_errors"] = [r for r in frows if "error" in r]
        by = {}
        for r in frows:
            if "seconds" in r:
                by.setdefault(r.get("framework"), []).append(r["seconds"])
        S["framework_seconds_median"] = {k: statistics.median(v) for k, v in by.items()}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--framework-seeds", type=int, default=2)
    ap.add_argument("--skip-frameworks", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, runtimes = engine_part(a.out, a.seeds)
    norm = no_rounding_baseline(a.out)
    frows = [] if a.skip_frameworks else framework_part(a.framework_seeds, a.out)
    S = summarize(rows, runtimes, norm, frows)
    with open(os.path.join(a.out, "engine_port_study.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": S, "rows": rows, "framework_rows": frows}, f, indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
