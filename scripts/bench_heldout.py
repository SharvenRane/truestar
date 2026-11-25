"""Held out kernel sets: `truestar kernel` against baselines on exactly the same input variants.

Sets (fetched by scripts/fetch_heldout.py; protocol and bounds in docs/KERNEL_CHECK.md):
  KB    KernelBench samples (generated kernels with KernelBench's published eval label)
  METR  METR KernelBenchFiltered best agent solutions (all published as passing METR's check)

Every kernel runs in its own process. `check_kernel` runs with its default variants; its `_call`
and `_judge` are wrapped here (in this script only) so that the very output each variant produced
is also judged by the baselines against the float64 reference:

  assert_close   torch.testing.assert_close defaults, equal_nan=True, float64 reference cast
  allclose_1e-2  torch.allclose(rtol=atol=1e-2) against the float64 reference
  allclose_1e-4  torch.allclose(rtol=atol=1e-4) against the float64 reference

and each method gets a verdict on three scopes: example (the example input only), task (example
plus value families at the task's shapes; truestar adds its repeat run) and all (adds the shape
and stride variants).

    python scripts/bench_heldout.py --list items.json
    python scripts/bench_heldout.py --run items.json --out-dir DIR [--workers 4] [--deadline 480]
    python scripts/bench_heldout.py --one '<item json>' --out res.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))  # works under python -I too
DATA =os.path.join(ROOT, "tests", "data", "heldout", "kernelbench")
METR = os.path.join(ROOT, "results", "heldout_metr")
METHODS = ("truestar", "assert_close", "allclose_1e-2", "allclose_1e-4")
SCOPES = {"example": ("base",), "task": ("base", "family", "repeat"),
          "all": ("base", "family", "repeat", "shape", "stride", "dtype", "extra")}
# Named as numerically wrong in the YAKK post (hud.ai, 2026-11-19), footnote 22 and the text.
YAKK_NAMED = {(1, 14), (1, 33), (1, 100), (2, 24), (2, 42)}


# ----------------------------------------------------------------------------- item list


def problem_path(lv, name):
    """Problem file for a KernelBench name. Long upstream names are stored shortened (Windows path
    limit); problems/NAMES.json maps level<lv>/<upstream name> to the stored name."""
    names = os.path.join(DATA, "problems", "NAMES.json")
    table = json.load(open(names, encoding="utf-8")) if os.path.isfile(names) else {}
    return os.path.join(DATA, "problems", f"level{lv}", table.get(f"level{lv}/{name}", name))


def list_items():
    items = []
    for p in sorted(glob.glob(os.path.join(DATA, "samples", "level*", "*", "problem_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        ev = list(d["eval_result"].values())[0] if d.get("eval_result") else {}
        if not ev.get("compiled"):
            continue
        lv = int(d["level"])
        model = os.path.basename(os.path.dirname(p))
        items.append({"id": f"KB-L{lv}-{model}-{d['problem_id']}", "set": "KB", "level": lv,
                      "problem": d["problem_name"],
                      "ref": os.path.relpath(problem_path(lv, d["problem_name"]), ROOT),
                      "cand_json": os.path.relpath(p, ROOT).replace("\\", "/"),
                      "label": "correct" if ev.get("correct") else "wrong",
                      "label_source": f"KernelBench eval {ev.get('correctness_trials', '')}"})
    for p in sorted(glob.glob(os.path.join(METR, "best_agent_solutions", "level_*", "task_*.py"))):
        lv = int(re.search(r"level_(\d+)", p).group(1))
        idx = int(re.search(r"task_(\d+)\.py", p).group(1))
        refs = glob.glob(os.path.join(METR, "tasks", f"level_{lv}", f"{idx}_*.py"))
        if len(refs) != 1:
            continue
        items.append({"id": f"METR-L{lv}-{idx}", "set": "METR", "level": lv,
                      "problem": os.path.basename(refs[0]),
                      "ref": os.path.relpath(refs[0], ROOT), "cand": os.path.relpath(p, ROOT).replace("\\", "/"),
                      "label": "wrong" if (lv, idx) in YAKK_NAMED else "correct",
                      "label_source": "named in the YAKK post" if (lv, idx) in YAKK_NAMED
                      else "passed METR's check"})
    for it in items:  # paths are used on Linux workers
        for k in ("ref", "cand", "cand_json"):
            if k in it:
                it[k] = it[k].replace("\\", "/")
    return items


# ----------------------------------------------------------------------------- one kernel


def _flat(o):
    from truestar.capture import flatten_tensors

    return list(flatten_tensors(o))


def _baselines(reference, out, inputs, device):
    import torch
    from truestar.tolerance import _fn64

    with torch.no_grad():
        ref = _fn64(reference, device)(*tuple(
            t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in inputs))
    outs, refs = _flat(out), _flat(ref)
    res = {m: "ok" for m in METHODS[1:]}
    if len(outs) != len(refs) or any(o.shape != r.shape for o, r in zip(outs, refs)):
        return {m: "fail" for m in res}, "output shape differs"
    num = {}  # numerics only: elements where float64 is finite (as truestar's TS202 policy)
    for o, r in zip(outs, refs):
        if not o.is_floating_point():
            if not torch.equal(o.cpu(), r.to(o.dtype).cpu()):
                res = {m: "fail" for m in res}
            continue
        _judge_baselines(o, r, res)
        fin = torch.isfinite(r)
        if not bool(fin.all()):
            num_one = {m: "ok" for m in METHODS[1:]}
            _judge_baselines(o[fin], r[fin], num_one)
            for m, x in num_one.items():
                if x == "fail":
                    num[m] = "fail"
                else:
                    num.setdefault(m, "ok")
    if num:
        res.update({m + "#num": num.get(m, "ok") for m in METHODS[1:]})
    return res, ""


def _judge_baselines(o, r, res):
    import torch

    try:
        torch.testing.assert_close(o, r.to(o.dtype), equal_nan=True)
    except AssertionError:
        res["assert_close"] = "fail"
    od = o.double()
    for m, tol in (("allclose_1e-2", 1e-2), ("allclose_1e-4", 1e-4)):
        if not bool(torch.allclose(od, r, rtol=tol, atol=tol, equal_nan=True)):
            res[m] = "fail"


def run_one(item, out_path):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    import torch
    from truestar import kernelcheck

    t0 = time.perf_counter()
    result = dict(item)
    result.update(status="error", variants=[], verdicts={}, codes=[], first_failure=None)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    result["device"] = str(torch.cuda.get_device_name(0)) if dev.type == "cuda" else "cpu"
    result["flags"] = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_tf32": torch.backends.cudnn.allow_tf32, "torch": torch.__version__}

    def dump():
        result["seconds"] = round(time.perf_counter() - t0, 2)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)

    try:
        ref_mod = kernelcheck._load(os.path.join(ROOT, item["ref"]), "ref")
        if "cand_json" in item:
            src = json.load(open(os.path.join(ROOT, item["cand_json"]), encoding="utf-8"))["kernel"]
            tmp = tempfile.mkdtemp(prefix="cand_")
            cand_path = os.path.join(tmp, "candidate_kernel.py")
            with open(cand_path, "w", encoding="utf-8") as f:
                f.write(src)
        else:
            cand_path = os.path.join(ROOT, item["cand"])
        tl = time.perf_counter()
        cand_mod = kernelcheck._load(cand_path, "cand")
        if item.get("compile_only"):
            result["status"] = "compiled"
            result["load_seconds"] = round(time.perf_counter() - tl, 2)
            dump()
            return 0
        init = ref_mod.get_init_inputs() if hasattr(ref_mod, "get_init_inputs") else []
        torch.manual_seed(0)
        reference = ref_mod.Model(*init).to(dev).eval()
        cls = getattr(cand_mod, "ModelNew", None) or getattr(cand_mod, "Model")
        torch.manual_seed(0)
        candidate = cls(*init).to(dev).eval()
        try:
            candidate.load_state_dict(reference.state_dict(), strict=False)
        except Exception:
            pass
        torch.manual_seed(0)
        inputs = ref_mod.get_inputs()
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
        inputs = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in inputs)
        result["load_seconds"] = round(time.perf_counter() - tl, 2)
    except BaseException as exc:
        result["status"] = "load_error"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        dump()
        return 0

    state = {}
    orig_call, orig_judge = kernelcheck._call, kernelcheck._judge

    def call(fn, ins, device):
        if fn is candidate:
            state["inputs"] = kernelcheck._clone(ins)
        out = orig_call(fn, ins, device)
        if fn is candidate:
            state["out"] = out
        return out

    def judge(ref_fn, cand_fn, v, device, report, base_ok, **kw):
        state.clear()
        res, rep = orig_judge(ref_fn, cand_fn, v, device, report, base_ok, **kw)
        entry = {"name": v.name, "kind": v.kind, "truestar": "skip" if res.status == "skipped"
                 else ("ok" if res.status == "ok" else ("warn" if res.status == "suspect"
                                                        else "fail")), "code": res.code,
                 "message": res.message[:300], "ratio": res.worst_ratio}
        if res.status == "skipped":
            entry.update({m: "skip" for m in METHODS[1:]})
        elif "out" not in state or res.code in ("TS201",):
            entry.update({m: "fail" for m in METHODS[1:]})
            entry["baseline_note"] = "candidate raised"
        else:
            try:
                b, note = _baselines(ref_fn, state["out"], state["inputs"], device)
                entry.update(b)
                if note:
                    entry["baseline_note"] = note
            except Exception as exc:
                entry.update({m: "error" for m in METHODS[1:]})
                entry["baseline_note"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        result["variants"].append(entry)
        state.clear()
        return res, rep

    kernelcheck._call, kernelcheck._judge = call, judge
    try:
        rep = kernelcheck.check_kernel(reference, candidate, inputs, device=dev, drill_ops=False,
                                       lean=True)
    except BaseException as exc:
        result["status"] = "check_error"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        result["trace"] = traceback.format_exc()[-1500:]
        dump()
        return 0
    finally:
        kernelcheck._call, kernelcheck._judge = orig_call, orig_judge
    for v in rep.variants:
        if v.kind == "repeat":
            result["variants"].append({"name": v.name, "kind": "repeat", "truestar": "warn",
                                       "code": v.code, "message": v.message[:300]})
    result["status"] = "done"
    result["truestar_status"] = rep.status
    result["codes"] = rep.codes
    result["first_failure"] = None if rep.first_failure is None else {
        "name": rep.first_failure.name, "code": rep.first_failure.code,
        "message": rep.first_failure.message[:500]}
    result["notes"] = rep.notes
    result["verdicts"] = verdicts(result["variants"])
    dump()
    return 0


def verdicts(variants):
    """{scope: {method: 'pass'|'reject'}} from per variant entries."""
    out = {}
    for scope, kinds in SCOPES.items():
        out[scope] = {}
        for m in METHODS:
            vs = [v.get(m) for v in variants if v["kind"] in kinds and v.get(m) is not None]
            out[scope][m] = "reject" if any(s in ("fail", "error") for s in vs) else "pass"
    return out


# ----------------------------------------------------------------------------- pool


def run_pool(items_path, out_dir, workers, deadline, timeout, only=None,
             redo=("load_error", "crash", "check_error"), compile_only=False):
    items = json.load(open(items_path, encoding="utf-8"))
    if compile_only:
        redo = ()
    if only:
        items = [i for i in items if re.search(only, i["id"])]
    os.makedirs(out_dir, exist_ok=True)
    def _st(i):
        p = os.path.join(out_dir, i["id"] + ".json")
        return json.load(open(p, encoding="utf-8")).get("status") if os.path.exists(p) else None
    todo = [i for i in items if _st(i) in (None,) + tuple(redo)]
    print(f"{len(items)} items, {len(todo)} to run, {workers} workers", flush=True)
    t_end = time.time() + deadline
    running = []
    done = 0
    while todo or running:
        while todo and len(running) < workers and time.time() + 60 < t_end:
            it = dict(todo.pop(0), compile_only=compile_only) if compile_only else todo.pop(0)
            out = os.path.join(out_dir, it["id"] + ".json")
            env = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(
                os.environ.get("HELDOUT_EXT_DIR", tempfile.gettempdir()), it["id"]),
                MAX_JOBS="2")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one",
                                  json.dumps(it), "--out", out], env=env,
                                 stdout=subprocess.DEVNULL, stderr=open(out + ".log", "w"))
            running.append((p, it, out, time.time()))
        time.sleep(0.5)
        for r in list(running):
            p, it, out, t0 = r
            late = time.time() - t0 > timeout or time.time() > t_end
            if p.poll() is None and not late:
                continue
            if p.poll() is None:
                p.kill()
                p.wait()
                if time.time() - t0 > timeout:
                    res = dict(it, status="timeout", seconds=round(time.time() - t0, 1))
                    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
            elif not os.path.exists(out):
                err = open(out + ".log", encoding="utf-8", errors="replace").read()[-1500:]
                res = dict(it, status="crash", returncode=p.returncode, error=err)
                json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
            running.remove(r)
            done += 1
            if os.path.exists(out):
                st = json.load(open(out, encoding="utf-8")).get("status")
                print(f"{done:4d} {it['id']:40s} {st}", flush=True)
        if not todo and not running:
            break
        if time.time() > t_end and not running:
            break
    left = len([i for i in items if not os.path.exists(os.path.join(out_dir, i["id"] + ".json"))])
    print(f"left: {left}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=None)
    ap.add_argument("--one", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default=None, help="regex on item ids")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=480)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        items = list_items()
        with open(a.list, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1)
        from collections import Counter

        print(Counter((i["set"], i["label"]) for i in items))
        return 0
    if a.one:
        return run_one(json.loads(a.one), a.out)
    if a.run:
        return run_pool(a.run, a.out_dir, a.workers, a.deadline, a.timeout, a.only,
                        compile_only=a.compile_only)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
