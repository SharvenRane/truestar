"""Tables for scripts/bench_heldout.py results: false fail and false pass per set, method, scope
and policy (protocol in docs/KERNEL_CHECK.md, held out section).

    python -I scripts/heldout_summary.py DIR[,DIR...] [--adjudicate] [--no-ts1301]

Policies. numerics: a method rejects a kernel for a wrong number only; truestar's TS202 (NaN
handled differently while every finite element is right) and TS203 (nondeterministic) verdicts do
not reject, and the baselines judge the NaN family on the elements where float64 is finite.
full: those verdicts reject too, and the baselines compare NaN positions (equal_nan).
`truestar+lean` is truestar with the TS1300 lean verdict on.

--adjudicate writes tests/data/heldout/adjudication.json, one label per kernel whose published
label it changes, with the reason:

  wrong               labelled correct, and the source uses the tanh approximation of GELU where
                      the task's GELU is the erf one; or on the example input or a finite value
                      family (not the NaN family) the candidate raised, gave infinity or NaN
                      where float64 is finite or an output of the wrong shape (allclose 1e-2
                      agreeing), or its error is at least
                      100x the derived float32 bound (orders of magnitude beyond a stable float32
                      algorithm); `label_strict` also asks for an error above 1e-2 (rtol and
                      atol) against float64, a rule that does not lean on truestar's bound
  precision_shedding  labelled correct, not wrong above, and the source computes in float16,
                      bfloat16 or TF32 (a source scan); wrong in the main table, correct in the
                      sensitivity table
  shape_limited       labelled correct, and the error rule above holds only on a shape or stride
                      variant: correct at example and task scope, wrong over all variants
  correct             labelled wrong (KernelBench), and every method passes every variant: the
                      label comes from KernelBench's harness (train mode, own weights)
"""
import collections
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADJ = os.path.join(ROOT, "tests", "data", "heldout", "adjudication.json")
BASELINES = ("assert_close", "allclose_1e-2", "allclose_1e-4")
METHODS = ("truestar", "truestar+lean") + BASELINES
SCOPES = {"example": ("base",), "task": ("base", "family", "repeat"),
          "all": ("base", "family", "repeat", "shape", "stride", "dtype", "extra")}
NAN_FAMILY = "NaN in one element"
FAR = 100.0  # times the derived bound


def _rejects(v, m, policy):
    if m.startswith("truestar"):
        s, code = v.get("truestar"), v.get("code")
        if code == "TS1300":
            return m == "truestar+lean"
        return s == "fail" or (policy == "full" and s == "warn")
    if v["kind"] == "repeat":
        return False
    s = v.get(m + "#num", v.get(m)) if policy == "numerics" else v.get(m)
    return s in ("fail", "error")


def verdict(r, scope, m, policy):
    return "reject" if any(_rejects(v, m, policy) for v in r["variants"]
                           if v["kind"] in SCOPES[scope]) else "pass"


def _source(it):
    if "cand_json" in it:
        return json.load(open(os.path.join(ROOT, it["cand_json"]), encoding="utf-8"))["kernel"]
    return open(os.path.join(ROOT, it["cand"]), encoding="utf-8").read()


def source_tags(r):
    src, ref = _source(r), open(os.path.join(ROOT, r["ref"]), encoding="utf-8").read()
    tags = set()
    if "0.044715" in src and "tanh" not in ref.lower():
        tags.add("tanh_gelu")
    if re.search(r"\.half\(\)|__half|at::Half|float16|__float2half|bfloat16|__nv_bfloat16", src):
        tags.add("low_precision")
    if "allow_tf32" in src and "True" in src:
        tags.add("tf32")
    return tags


def _far(v, strict):
    if v.get("code") == "TS201":
        return True
    msg = v.get("message") or ""
    if v.get("truestar") == "fail" and v.get("allclose_1e-2") == "fail" and (
            msg.startswith("output: NaN or infinity") or " vs float64 reference " in msg):
        return True  # infinity or NaN where float64 is finite, or the wrong output shape
    return (v.get("ratio") or 0) >= FAR and (not strict or v.get("allclose_1e-2") == "fail")


def _label(r, strict):
    vs = r["variants"]
    fin = [v for v in vs if v["kind"] in ("base", "family") and v["name"] != NAN_FAMILY]
    other = [v for v in vs if v["kind"] in ("shape", "stride")]
    if r["label"] == "correct":
        tags = source_tags(r)
        if "tanh_gelu" in tags:
            return "wrong", "the source uses the tanh approximation of GELU; the task's is erf"
        hit = next((v for v in fin if _far(v, strict)), None)
        if hit:
            return "wrong", f"{hit['name']}: {(hit.get('message') or '')[:160]}"
        if tags & {"low_precision", "tf32"}:
            return "precision_shedding", "the source computes in " + " and ".join(sorted(tags))
        hit = next((v for v in other if _far(v, strict)), None)
        if hit:
            return "shape_limited", f"{hit['name']}: {(hit.get('message') or '')[:160]}"
    elif r["set"] == "KB" and all(verdict(r, "all", m, "full") == "pass"
                                  for m in ("truestar",) + BASELINES):
        return "correct", ("every method passes every variant; KernelBench's label comes from its "
                           "harness (train mode, own weights)")
    return r["label"], ""


def adjudicate(rows):
    adj = {}
    for r in rows:
        lab, why = _label(r, False)
        lab_s, _ = _label(r, True)
        if lab != r["label"] or lab_s != r["label"]:
            adj[r["id"]] = {"label": lab, "label_strict": lab_s, "reason": why}
    return adj


def truth(r, adj, scope, shed_correct=False, key="label"):
    cat = adj.get(r["id"], {}).get(key, r["label"])
    if cat == "precision_shedding":
        return "correct" if shed_correct else "wrong"
    if cat == "shape_limited":
        return "wrong" if scope == "all" else "correct"
    return cat


def table(rows, adj, policy, shed_correct=False, key="label"):
    out = []
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == s]
        for scope in SCOPES:
            tr = [truth(r, adj, scope, shed_correct, key) for r in rs]
            n_ok = tr.count("correct")
            n_bad = len(rs) - n_ok
            for m in METHODS:
                vd = [verdict(r, scope, m, policy) for r in rs]
                ff = sum(t == "correct" and x == "reject" for t, x in zip(tr, vd))
                fp = sum(t == "wrong" and x == "pass" for t, x in zip(tr, vd))
                out.append((s, scope, m, ff, n_ok, fp, n_bad))
    return out


def main(argv):
    # Several result folders, comma separated: a kernel's result in a later folder replaces it,
    # except a later result that did not finish replaces a finished one.
    by_id = {}
    for d in argv[0].split(","):
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            r = json.load(open(p, encoding="utf-8"))
            old = by_id.get(r["id"])
            if old is None or r.get("status") == "done" or old.get("status") != "done":
                by_id[r["id"]] = r
    res = [by_id[k] for k in sorted(by_id)]
    for r in res:  # before the fix, a float64 reference out of GPU memory read as a skip
        if r.get("status") == "done" and any(
                v.get("truestar") == "skip" and "rejects it (OutOfMemoryError" in
                (v.get("message") or "") for v in r["variants"]):
            r["status"] = "reference_out_of_memory"
    rows = [r for r in res if r.get("status") == "done"]
    for r in rows if "--no-ts1301" not in argv else []:  # runs made before TS1301 existed:
        for v in r["variants"]:  # the same source scan, applied here
            if v.get("code") == "TS1300" and source_tags(r) & {"low_precision", "tf32"}:
                v.update(code="TS1301", truestar="warn", offline="TS1301 from the source scan")
    if "--adjudicate" in argv:
        with open(ADJ, "w", encoding="utf-8") as f:
            json.dump(adjudicate(rows), f, indent=1, sort_keys=True)
    adj = json.load(open(ADJ, encoding="utf-8")) if os.path.exists(ADJ) else {}
    print("status:", dict(sorted(collections.Counter((r["set"], r.get("status"))
                                                     for r in res).items())))
    for key in ("label", "label_strict"):
        print(f"adjudicated ({key}):", dict(sorted(collections.Counter(
            (r["set"], adj[r["id"]][key]) for r in rows if r["id"] in adj).items())))
    runs = [("adjudicated labels", adj, "label", False),
            ("adjudicated labels, precision shedding counted correct", adj, "label", True),
            ("adjudicated labels, strict rule", adj, "label_strict", False),
            ("published labels", {}, "label", False)]
    for policy in ("numerics", "full"):
        for title, labels, key, shed in runs:
            print(f"\n## policy {policy}, {title}")
            print(f"{'set':5s} {'scope':8s} {'method':14s} {'false fail':>16s} {'false pass':>16s}")
            for s, scope, m, ff, n_ok, fp, n_bad in table(rows, labels, policy, shed, key):
                print(f"{s:5s} {scope:8s} {m:14s} {ff:4d}/{n_ok:<4d} {100 * ff / max(1, n_ok):5.1f}% "
                      f"{fp:4d}/{n_bad:<4d} {100 * fp / max(1, n_bad):5.1f}%")
    for code in ("TS1300", "TS1301"):
        hit = [(r["id"], adj.get(r["id"], {}).get("label", r["label"])) for r in rows
               if any(v.get("code") == code for v in r["variants"])]
        print(f"\n{code} on {len(hit)} kernels (id, adjudicated label): {hit}")
    secs = sorted(r["seconds"] - r.get("load_seconds", 0) for r in rows)
    if secs:
        print(f"check seconds per kernel (all methods, L4): median {secs[len(secs) // 2]:.1f}, "
              f"max {secs[-1]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
