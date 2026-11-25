"""Diff two fingerprint JSON files: which settings differ, which rows are bitwise equal.

A row is (case, dtype, arm, batch). Its status is ``equal`` (every output hash equal),
``different``, ``not_comparable`` (the two files did not start from the same weights or inputs),
``error`` (one side failed) or ``only_a`` / ``only_b``. Only ``truestar_det`` rows carry a claim;
a different one is TS743 and the comparison exits 1. Every other difference is reported as data.

Files from the first cross chip script (``{"env": ..., "results": [...]}``, cases mlp, transformer
and resnet18 at batch 4) are read too: their cases become ``legacy_*`` with the arms
``truestar_det`` (its ``compiled_cuda``) and ``eager`` (its ``eager_cuda``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .run import SCHEMA

ENV_KEYS = ("gpu", "capability", "sm_count", "gpu_memory_mib", "driver", "cuda_runtime", "cudnn", "torch",
            "triton", "torchvision", "numpy", "python", "platform", "cpu", "truestar")


class FingerprintFileError(Exception):
    """Not a fingerprint file (TS749)."""


def _legacy(doc: dict) -> dict:
    env = dict(doc.get("env", {}))
    env["cuda_runtime"] = env.pop("driver_cuda", None)
    env.setdefault("driver", None)
    env["git"] = {"commit": None, "dirty": None, "source": "not recorded (first cross chip script)"}
    out = {"schema": SCHEMA, "legacy_format": True, "env": env,
           "settings": {"note": "first cross chip script: float32, TF32 off, deterministic algorithms on, "
                                "compiled with deterministic=True, allow_tf32=False"},
           "data": {}, "groups": []}
    settings = {"torch_deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                "matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    for r in doc.get("results", []):
        case = "legacy_" + r["case"]
        batch = r.get("input_shape", [None])[0]
        out["data"][case] = {"weights_sha256": r.get("weights_sha256"),
                             "input_tensor_sha256": [r.get("input_sha256")],
                             "input_shapes": [r.get("input_shape")], "legacy": True}
        for arm, key in (("truestar_det", "compiled_cuda"), ("eager", "eager_cuda")):
            g = {"case": case, "dtype": "float32", "arm": arm, "claim": "bitwise" if arm == "truestar_det" else "none",
                 "settings": settings, "truestar_deterministic": True if arm == "truestar_det" else None,
                 "rows": [], "error": None}
            if key in r:
                g["rows"].append({"batch": batch, "outputs": r[key], "again": r.get(key + "_again"),
                                  "run_to_run_equal": r.get(key) == r.get(key + "_again")})
            else:
                g["error"] = r.get("compiled_error", "not recorded")
            out["groups"].append(g)
    return out


def load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintFileError(f"{path}: cannot read a JSON document ({exc})") from exc
    if not isinstance(doc, dict):
        raise FingerprintFileError(f"{path}: not a fingerprint (the JSON is not an object)")
    if doc.get("schema") == SCHEMA:
        return doc
    if "env" in doc and isinstance(doc.get("results"), list) and all("case" in r for r in doc["results"]):
        return _legacy(doc)
    raise FingerprintFileError(f"{path}: not a fingerprint (schema {doc.get('schema')!r}, expected {SCHEMA!r})")


def _diff_dicts(a: dict, b: dict, prefix: str = "") -> List[dict]:
    out = []
    for k in sorted(set(a) | set(b), key=str):
        va, vb = a.get(k, "<not recorded>"), b.get(k, "<not recorded>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff_dicts(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append({"key": f"{prefix}{k}", "a": va, "b": vb})
    return out


def _configs(g: dict) -> Dict[Tuple[str, str], dict]:
    return {(c["step"], json.dumps(c["shapes"])): c for c in g.get("configs", [])}


@dataclass
class Comparison:
    a_name: str
    b_name: str
    a_env: dict
    b_env: dict
    env_diff: List[dict] = field(default_factory=list)
    settings_diff: List[dict] = field(default_factory=list)
    data: List[dict] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    issues: List[dict] = field(default_factory=list)

    @property
    def claim_ok(self) -> bool:
        return not any(i["code"] in ("TS743",) for i in self.issues)

    def summary(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.rows:
            s = out.setdefault(r["arm"], {})
            s[r["status"]] = s.get(r["status"], 0) + 1
        return out

    def as_dict(self) -> dict:
        return {"a": {"file": self.a_name, **{k: self.a_env.get(k) for k in ENV_KEYS},
                      "commit": (self.a_env.get("git") or {}).get("commit")},
                "b": {"file": self.b_name, **{k: self.b_env.get(k) for k in ENV_KEYS},
                      "commit": (self.b_env.get("git") or {}).get("commit")},
                "claim_ok": self.claim_ok, "env_diff": self.env_diff, "settings_diff": self.settings_diff,
                "data": self.data, "summary": self.summary(), "rows": self.rows, "issues": self.issues}

    def to_text(self) -> str:
        def who(env):
            return (f"{env.get('gpu')} (cc {env.get('capability')}, {env.get('sm_count')} SMs), "
                    f"torch {env.get('torch')}, Triton {env.get('triton')}, CUDA {env.get('cuda_runtime')}, "
                    f"driver {env.get('driver')}")
        lines = [f"A: {self.a_name}: {who(self.a_env)}", f"B: {self.b_name}: {who(self.b_env)}", ""]
        lines.append("environment differs in: " + (", ".join(d["key"] for d in self.env_diff) or "nothing"))
        for d in self.env_diff:
            lines.append(f"  {d['key']}: A {d['a']!r}  B {d['b']!r}")
        lines.append("settings differ in: " + (", ".join(sorted({d['key'] for d in self.settings_diff})) or "nothing"))
        for d in self.settings_diff[:40]:
            lines.append(f"  {d.get('where', '')} {d['key']}: A {d['a']!r}  B {d['b']!r}".rstrip())
        bad = [d for d in self.data if not d["weights_equal"] or not d["inputs_equal"]]
        lines.append(f"weights and inputs: {len(self.data) - len(bad)} of {len(self.data)} cases start from the same bits")
        for d in bad:
            lines.append(f"  {d['case']}: weights equal {d['weights_equal']}, inputs equal {d['inputs_equal']}")
        lines.append("")
        for arm, s in self.summary().items():
            lines.append(f"{arm:<17} " + ", ".join(f"{k} {v}" for k, v in sorted(s.items())))
        lines.append("")
        for r in self.rows:
            if r["status"] == "equal":
                continue
            extra = ""
            if r.get("configs_differ"):
                extra = f"; launch configs differ in {len(r['config_diffs'])} kernels"
            if r.get("library_ops"):
                extra += f"; library calls {', '.join(r['library_ops'][:4])}"
            if r.get("error"):
                extra += f"; {r['error'][:140]}"
            lines.append(f"  {r['case']:<18} {r['dtype']:<8} {r['arm']:<16} batch {str(r['batch']):>3}: {r['status']}{extra}")
        lines.append("")
        for i in self.issues:
            lines.append(f"[{i['code']}] {i['message']}")
        lines.append("claim (truestar_det bitwise equal on both): " + ("holds" if self.claim_ok else "BROKEN"))
        return "\n".join(lines)


def compare(a: dict, b: dict, a_name: str = "A", b_name: str = "B") -> Comparison:
    ea, eb = a.get("env", {}), b.get("env", {})
    cmp = Comparison(a_name, b_name, ea, eb)
    cmp.env_diff = [{"key": k, "a": ea.get(k), "b": eb.get(k)} for k in ENV_KEYS if ea.get(k) != eb.get(k)]
    ga, gb = (ea.get("git") or {}), (eb.get("git") or {})
    if ga.get("commit") != gb.get("commit"):
        cmp.env_diff.append({"key": "git.commit", "a": ga.get("commit"), "b": gb.get("commit")})
    cmp.settings_diff = [dict(d, where="global") for d in _diff_dicts(a.get("settings", {}), b.get("settings", {}))]
    if a.get("fresh_cache") != b.get("fresh_cache") and not (a.get("legacy_format") or b.get("legacy_format")):
        cmp.settings_diff.append({"where": "global", "key": "fresh_cache", "a": a.get("fresh_cache"),
                                  "b": b.get("fresh_cache")})

    same_data = {}
    for case in sorted(set(a.get("data", {})) & set(b.get("data", {}))):
        da, db = a["data"][case], b["data"][case]
        w = da.get("weights_sha256") == db.get("weights_sha256")
        i = da.get("input_tensor_sha256") == db.get("input_tensor_sha256")
        same_data[case] = w and i
        cmp.data.append({"case": case, "weights_equal": w, "inputs_equal": i})
        if not (w and i):
            cmp.issues.append({"code": "TS744", "message": f"{case}: the two runs started from different "
                                                           f"{'weights' if not w else 'inputs'}; its outputs are not comparable"})

    groups_a = {(g["case"], g["dtype"], g["arm"]): g for g in a.get("groups", [])}
    groups_b = {(g["case"], g["dtype"], g["arm"]): g for g in b.get("groups", [])}
    settings_seen = set()
    for key in sorted(set(groups_a) | set(groups_b)):
        g1, g2 = groups_a.get(key), groups_b.get(key)
        case, dtype, arm = key
        base = {"case": case, "dtype": dtype, "arm": arm}
        if g1 is None or g2 is None:
            g = g1 or g2
            for r in g["rows"]:
                cmp.rows.append(dict(base, batch=r["batch"], claim=g["claim"], status="only_a" if g2 is None else "only_b"))
            continue
        claim = "bitwise" if "bitwise" in (g1.get("claim"), g2.get("claim")) else "none"
        for d in _diff_dicts(g1.get("settings", {}), g2.get("settings", {})):
            if (arm, dtype, d["key"]) not in settings_seen:
                settings_seen.add((arm, dtype, d["key"]))
                cmp.settings_diff.append(dict(d, where=f"{arm} {dtype}"))
        if g1.get("truestar_deterministic") != g2.get("truestar_deterministic"):
            cmp.settings_diff.append({"where": f"{arm} {dtype}", "key": "truestar_deterministic",
                                      "a": g1.get("truestar_deterministic"), "b": g2.get("truestar_deterministic")})
        c1, c2 = _configs(g1), _configs(g2)
        config_diffs = []
        for k in sorted(set(c1) | set(c2)):
            x, y = c1.get(k), c2.get(k)
            if x is None or y is None or x["config"] != y["config"]:
                config_diffs.append({"step": k[0], "shapes": json.loads(k[1]),
                                     "a": x and {"config": x["config"], "source": x["source"]},
                                     "b": y and {"config": y["config"], "source": y["source"]}})
        lib = sorted(set((g1.get("compile") or {}).get("library_ops", []))
                     | set((g2.get("compile") or {}).get("library_ops", [])))
        rows1 = {r["batch"]: r for r in g1["rows"]}
        rows2 = {r["batch"]: r for r in g2["rows"]}
        for n in sorted(set(rows1) | set(rows2), key=lambda v: (v is None, v)):
            r1, r2 = rows1.get(n), rows2.get(n)
            row = dict(base, batch=n, claim=claim)
            if r1 is None or r2 is None:
                row["status"] = "only_a" if r2 is None else "only_b"
            elif g1.get("error") or g2.get("error") or "error" in r1 or "error" in r2:
                row["status"] = "error"
                row["error"] = "; ".join(f"{side}: {e}" for side, e in
                                         (("A", g1.get("error") or r1.get("error")),
                                          ("B", g2.get("error") or r2.get("error"))) if e)
            elif not same_data.get(case, True):
                row["status"] = "not_comparable"
            else:
                eq = r1.get("outputs") == r2.get("outputs")
                row["status"] = "equal" if eq else "different"
                if not eq:
                    row["outputs_equal"] = [x == y for x, y in zip(r1.get("outputs") or [], r2.get("outputs") or [])]
                    row["configs_differ"] = bool(config_diffs)
                    row["config_diffs"] = config_diffs
                    row["library_ops"] = lib
            cmp.rows.append(row)
            if row["status"] == "different" and claim == "bitwise":
                cmp.issues.append({"code": "TS743", "message": f"{case} {dtype} {arm} batch {n}: deterministic "
                                                               f"mode gave different bits on the two machines"})
        if g1.get("error") or g2.get("error"):
            g = g1 if g1.get("error") else g2
            cmp.issues.append({"code": "TS742", "message": f"{case} {dtype} {arm} failed on "
                                                           f"{'A' if g is g1 else 'B'}: {str(g['error'])[:200]}"})
    if cmp.env_diff:
        cmp.issues.append({"code": "TS747", "message": "environment differs: " + ", ".join(d["key"] for d in cmp.env_diff)})
    if cmp.settings_diff:
        cmp.issues.append({"code": "TS745", "message": "settings differ: "
                           + ", ".join(sorted({f"{d['where']} {d['key']}" for d in cmp.settings_diff}))[:600]})
    unclaimed = sum(1 for r in cmp.rows if r["status"] == "different" and r["claim"] != "bitwise")
    if unclaimed:
        cmp.issues.append({"code": "TS746", "message": f"{unclaimed} rows of arms without a claim (eager, "
                                                       f"inductor, truestar_default) differ; recorded, not a finding"})
    only = sum(1 for r in cmp.rows if r["status"] in ("only_a", "only_b"))
    if only:
        cmp.issues.append({"code": "TS748", "message": f"{only} rows are in one file only"})
    return cmp


def compare_files(path_a: str, path_b: str) -> Comparison:
    return compare(load(path_a), load(path_b), path_a, path_b)


def match_table(docs: List[Tuple[str, dict]], arm: str = "truestar_det") -> List[dict]:
    """For several fingerprints: per (case, dtype, batch) of ``arm``, the output hash prefix on each,
    used to build the table in docs/CROSS_CHIP.md."""
    keys: Dict[Tuple, Dict[str, Optional[str]]] = {}
    for name, doc in docs:
        for g in doc.get("groups", []):
            if g["arm"] != arm:
                continue
            for r in g["rows"]:
                k = (g["case"], g["dtype"], r["batch"])
                cell = "error" if (g.get("error") or "error" in r) else ",".join(h[:8] for h in r["outputs"])
                keys.setdefault(k, {})[name] = cell
    out = []
    for (case, dtype, batch), cells in sorted(keys.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        vals = [v for v in cells.values() if v not in (None, "error")]
        out.append({"case": case, "dtype": dtype, "batch": batch, "cells": cells,
                    "all_equal": len(vals) == len(docs) and len(set(vals)) == 1})
    return out
