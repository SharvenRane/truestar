"""`truestar conform vllm DIR` / `truestar conform sglang DIR`: judge what a GPU serving engine computed.

DIR is what `serving_battery.py` wrote where the engine ran. This side needs no GPU and no engine: it
rebuilds the weights and prompts from the saved spec, checks their hashes against the run, and
computes the float64 reference of the same weights itself (docs/CONFORM_SERVING.md).

Against float64, per pass:

* every prompt position and every generated position: |engine logprob - float64 logprob| of the
  token at that position, against the derived tolerance. The tolerance is 10 times the RMS error a
  correct CPU run in the serving dtype shows on the same sequences, and at least a floor per dtype.
* decisions: the engine's top 1 at a prompt position and its greedy token at a generated position,
  against the float64 argmax given the same tokens (the reference is teacher forced on the engine's
  own generation, so one early difference does not hide later ones). A different choice is a near tie
  (ambiguous, either is correct) when the float64 margin between the two is at most twice the largest
  error the correct serving dtype run shows on any position; otherwise it is a real divergence.

Against itself: the same requests twice, one prompt at a time against eight in one batch, prefix
caching on against off, chunked prefill on against off. Reported bit for bit, with the largest
logprob change and the first generated token that changed.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .serving_battery import FORMAT, SETTINGS
from .serving_model import DTYPES, ServeSpec, logprobs, prompts, weights, weights_sha256

FLOOR = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}  # nats
TOL_FACTOR = 10.0
TIE_FACTOR = 2.0

SELF_CHECKS = [  # (key, code, label, (setting, pass) baseline, (setting, pass) candidate)
    ("repeat", "TS1326", "same request twice", ("base", "batch1"), ("base", "batch1_repeat")),
    ("batch", "TS1327", "batch 1 vs batch 8", ("base", "batch1"), ("base", "batch8")),
    ("prefix_cache", "TS1328", "prefix caching off vs on", ("base", "batch1"), ("prefix_cache", "batch1")),
    ("prefix_cache_gen", "TS1328", "prefix caching off vs on, generation only", ("base", "gen_only"),
     ("prefix_cache", "gen_only")),
    ("chunked_prefill", "TS1329", "chunked prefill off vs on", ("base", "batch1"), ("chunked_prefill", "batch1")),
]


class ServingBundleError(Exception):
    """The results directory cannot be judged (TS1322)."""


@dataclass
class PassVerdict:
    setting: str
    name: str
    positions: int = 0
    over_tol: int = 0
    rms: float = 0.0               # RMS logprob error over the positions
    worst: Optional[Dict[str, Any]] = None
    real_flips: int = 0            # prompt top 1 differs, not a near tie
    tie_flips: int = 0
    greedy_real: int = 0           # generated tokens that differ from float64 argmax, not a near tie
    greedy_tie: int = 0
    first_real: Optional[Dict[str, Any]] = None
    first_tie: Optional[Dict[str, Any]] = None
    missing: int = 0               # positions the engine returned no logprob for

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SelfVerdict:
    key: str
    code: str
    label: str
    baseline: str
    candidate: str
    available: bool = True
    reason: Optional[str] = None
    bitwise: bool = True
    max_dlogprob: float = 0.0
    generations_changed: int = 0
    first_change: Optional[Dict[str, Any]] = None
    top1_changed: int = 0
    cached_tokens: Optional[int] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ServingReport:
    engine: str
    results_dir: str
    meta: Dict[str, Any]
    tol: float
    emulated_rms: float
    emulated_max: float
    tie: float
    floor: float
    passes: List[PassVerdict]
    selfs: List[SelfVerdict]
    setting_errors: Dict[str, str]
    issues: List[Tuple[str, str]]

    @property
    def finding(self) -> bool:
        from ..codes import severity

        return any(severity(c) != "info" for c, _ in self.issues)

    def as_dict(self) -> Dict[str, Any]:
        m = self.meta
        return {"engine": self.engine, "engine_version": m.get("engine_version"), "gpu": m.get("gpu"),
                "dtype": m["spec"]["dtype"], "results_dir": self.results_dir,
                "tolerance": {"nats": self.tol, "emulated_rms": self.emulated_rms, "floor": self.floor,
                              "emulated_max": self.emulated_max, "near_tie_margin": self.tie,
                              "factor": TOL_FACTOR, "tie_factor": TIE_FACTOR},
                "settings": {s: {"flags": (m.get("_runs_flags") or {}).get(s),
                                 "error": self.setting_errors.get(s)} for s in m.get("settings", [])},
                "vs_float64": [p.as_dict() for p in self.passes],
                "vs_itself": [s.as_dict() for s in self.selfs]}


def load(results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mp, rp = os.path.join(results_dir, "meta.json"), os.path.join(results_dir, "runs.json")
    if not os.path.isfile(rp) and os.path.isfile(rp + ".gz"):
        rp += ".gz"
    if not os.path.isdir(results_dir):
        raise ServingBundleError(f"{results_dir} is not a directory")
    for p in (mp, rp):
        if not os.path.isfile(p):
            raise ServingBundleError(f"{p} is missing; write DIR with "
                                     "`python -m truestar.conform.serving_battery run --engine ENGINE --out DIR`")
    try:
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
        with (gzip.open(rp, "rt", encoding="utf-8") if rp.endswith(".gz") else open(rp, encoding="utf-8")) as f:
            runs = json.load(f)
    except (OSError, ValueError) as exc:
        raise ServingBundleError(f"{results_dir}: unreadable results ({type(exc).__name__}: {exc})") from exc
    if meta.get("format") != FORMAT:
        raise ServingBundleError(f"{mp} is not a {FORMAT} file")
    if meta.get("spec", {}).get("dtype") not in DTYPES:
        raise ServingBundleError(f"{mp}: unknown serving dtype {meta.get('spec', {}).get('dtype')!r}")
    return meta, runs


class _Ref:
    """float64 (and serving dtype) log probabilities per token sequence, computed once each."""

    def __init__(self, spec: ServeSpec):
        self.spec = spec
        self.W = weights(spec)
        self._c: Dict[Tuple[str, Tuple[int, ...]], torch.Tensor] = {}

    def lp(self, seq: Sequence[int], kind: str = "f64") -> torch.Tensor:
        key = (kind, tuple(seq))
        if key not in self._c:
            dt = torch.float64 if kind == "f64" else DTYPES[self.spec.dtype]
            self._c[key] = logprobs(self.spec, self.W, seq, dt)
        return self._c[key]


def _positions(p: List[int], rec: Dict[str, Any]):
    """(kind, index, row in seq, token, engine logprob, engine top1) for every judged position."""
    seq = list(p) + list(rec["tokens"])
    out = []
    for j, e in enumerate(rec.get("prompt") or []):
        if j == 0 or e is None:
            continue
        top1 = e["top"][0][0] if e.get("top") else None
        out.append(("prompt", j, j - 1, p[j], e.get("logprob"), top1))
    for k, e in enumerate(rec.get("gen") or []):
        out.append(("gen", k, len(p) - 1 + k, seq[len(p) + k], e.get("logprob"), seq[len(p) + k]))
    return seq, out


def _tolerance(spec: ServeSpec, P: List[List[int]], recs: List[Dict[str, Any]], ref: _Ref):
    errs = []
    for p, rec in zip(P, recs):
        seq, pos = _positions(p, rec)
        a, b = ref.lp(seq), ref.lp(seq, "emu")
        errs += [abs(float(b[row, tok] - a[row, tok])) for _, _, row, tok, _, _ in pos]
    rms = math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))
    worst = max(errs, default=0.0)
    return max(FLOOR[spec.dtype], TOL_FACTOR * rms), rms, worst


def judge_pass(setting: str, name: str, P: List[List[int]], recs: List[Dict[str, Any]], ref: _Ref,
               tol: float, tol_tie: float) -> PassVerdict:
    v = PassVerdict(setting, name)
    sq = 0.0
    for i, (p, rec) in enumerate(zip(P, recs)):
        seq, pos = _positions(p, rec)
        a = ref.lp(seq)
        for kind, idx, row, tok, elp, top1 in pos:
            if elp is None:
                v.missing += 1
                continue
            v.positions += 1
            err = abs(elp - float(a[row, tok]))
            sq += err * err
            if err > tol:
                v.over_tol += 1
            if v.worst is None or err > v.worst["error"]:
                v.worst = {"prompt": i, "kind": kind, "index": idx, "error": err, "engine": elp,
                           "float64": float(a[row, tok])}
            if top1 is None:
                continue
            best = int(a[row].argmax())
            if top1 == best:
                continue
            margin = float(a[row, best] - a[row, top1])
            tie = margin <= tol_tie
            where = {"prompt": i, "kind": kind, "index": idx, "engine_token": int(top1),
                     "float64_token": best, "float64_margin": margin}
            if kind == "prompt":
                if tie:
                    v.tie_flips += 1
                else:
                    v.real_flips += 1
            elif tie:
                v.greedy_tie += 1
                if v.first_tie is None or (idx, i) < (v.first_tie["index"], v.first_tie["prompt"]):
                    v.first_tie = where
            else:
                v.greedy_real += 1
                if v.first_real is None or (idx, i) < (v.first_real["index"], v.first_real["prompt"]):
                    v.first_real = where
    v.rms = math.sqrt(sq / v.positions) if v.positions else 0.0
    return v


def judge_self(key, code, label, base, cand, runs, ref: _Ref, P, tie) -> SelfVerdict:
    bs, bp = base
    cs, cp = cand
    v = SelfVerdict(key, code, label, f"{bs}.{bp}", f"{cs}.{cp}")
    A = (runs.get(bs) or {}).get("passes", {}).get(bp)
    B = (runs.get(cs) or {}).get("passes", {}).get(cp)
    if not A or not B:
        v.available = False
        v.reason = "; ".join(f"{s}: {runs[s]['error']}" for s in {bs, cs} if s in runs and runs[s].get("error")) \
            or "; ".join(f"setting {s} was not run" for s in (bs, cs) if s not in runs) \
            or "pass missing from the results"
        return v
    cached = [r.get("cached_tokens") for r in B]
    if any(c is not None for c in cached):
        v.cached_tokens = int(sum(c or 0 for c in cached))
    for i, (p, a, b) in enumerate(zip(P, A, B)):
        if a["tokens"] != b["tokens"]:
            v.generations_changed += 1
            k = next((k for k, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y),
                     min(len(a["tokens"]), len(b["tokens"])))
            row = len(p) - 1 + k
            r = ref.lp(list(p) + a["tokens"][:k])  # the context both runs share up to k
            ta = a["tokens"][k] if k < len(a["tokens"]) else None
            tb = b["tokens"][k] if k < len(b["tokens"]) else None
            margin = abs(float(r[row, ta] - r[row, tb])) if ta is not None and tb is not None else None
            ch = {"prompt": i, "index": k, "baseline_token": ta, "candidate_token": tb,
                  "float64_margin": margin, "near_tie": margin is not None and margin <= tie}
            if v.first_change is None or (k, i) < (v.first_change["index"], v.first_change["prompt"]):
                v.first_change = ch
        else:
            k = len(a["tokens"])
        pa, pb = a.get("prompt") or [], b.get("prompt") or []
        for j, (x, y) in enumerate(zip(pa, pb)):
            if x is None or y is None:
                continue
            if x["logprob"] is not None and y["logprob"] is not None:
                d = abs(x["logprob"] - y["logprob"])
                v.max_dlogprob = max(v.max_dlogprob, d)
            if x != y:
                v.bitwise = False
            if x.get("top") and y.get("top") and x["top"][0][0] != y["top"][0][0]:
                v.top1_changed += 1
        for x, y in zip(a["gen"][:k], b["gen"][:k]):  # same context only
            if x["logprob"] is not None and y["logprob"] is not None:
                v.max_dlogprob = max(v.max_dlogprob, abs(x["logprob"] - y["logprob"]))
            if x != y:
                v.bitwise = False
        if a["tokens"] != b["tokens"]:
            v.bitwise = False
    if key.startswith("prefix_cache") and v.cached_tokens == 0:
        v.notes.append("prefix caching was on but the engine reported 0 cached tokens for these requests: "
                       "this comparison did not exercise a cache read")
    if key == "prefix_cache" and v.cached_tokens is None:
        v.notes.append("the engine reports no cached token count, so a cache read cannot be confirmed")
    return v


def check(results_dir: str, engine: Optional[str] = None) -> ServingReport:
    meta, runs = load(results_dir)
    if engine and meta.get("engine") != engine:
        raise ServingBundleError(f"{results_dir} holds a {meta.get('engine')} run, not {engine}")
    spec = ServeSpec.from_dict(meta["spec"])
    ref = _Ref(spec)
    sha = weights_sha256(ref.W)
    if sha != meta.get("weights_sha256"):
        raise ServingBundleError(f"the weights rebuilt here (sha256 {sha[:12]}) are not the weights the engine "
                                 f"served ({str(meta.get('weights_sha256'))[:12]}); the spec or the generator differs")
    P = prompts(spec)
    if P != meta.get("prompts"):
        raise ServingBundleError("the prompts rebuilt here differ from the prompts the engine was sent")
    issues: List[Tuple[str, str]] = []
    errors = {s: r["error"] for s, r in runs.items() if r.get("error")}
    for s, e in errors.items():
        issues.append(("TS1322", f"setting {s} did not run in {meta['engine']}: {e[:400]}"))
    meta["_runs_flags"] = {s: r.get("flags") for s, r in runs.items()}
    base = (runs.get("base") or {}).get("passes", {}).get("batch1")
    if not base:
        raise ServingBundleError(f"the base setting has no results ({errors.get('base', 'missing')}); nothing "
                                 "to judge")
    tol, rms, emax = _tolerance(spec, P, base, ref)
    tie = TIE_FACTOR * max(FLOOR[spec.dtype], emax)
    passes = []
    for s, r in runs.items():
        for name, recs in (r.get("passes") or {}).items():
            if len(recs) != len(P):
                issues.append(("TS1322", f"{s}.{name}: {len(recs)} results for {len(P)} prompts"))
                continue
            passes.append(judge_pass(s, name, P, recs, ref, tol, tie))
    for pv in passes:
        tag = f"{pv.setting}.{pv.name}"
        if pv.over_tol:
            w = pv.worst
            issues.append(("TS1323", f"{tag}: {pv.over_tol} of {pv.positions} logprobs off float64 by more than "
                                     f"{tol:.3g} nats; worst {w['error']:.3g} at prompt {w['prompt']} "
                                     f"{w['kind']} position {w['index']}"))
        if pv.real_flips or pv.greedy_real:
            fr = pv.first_real
            first = (f"; first greedy divergence at prompt {fr['prompt']} token {fr['index']} "
                     f"(engine {fr['engine_token']}, float64 {fr['float64_token']}, margin {fr['float64_margin']:.3g})"
                     if fr else "")
            issues.append(("TS1324", f"{tag}: {pv.real_flips} prompt top 1 and {pv.greedy_real} greedy tokens "
                                     f"differ from float64 by more than a near tie{first}"))
        if pv.tie_flips or pv.greedy_tie:
            issues.append(("TS1325", f"{tag}: {pv.tie_flips} prompt top 1 and {pv.greedy_tie} greedy tokens "
                                     f"differ from float64 at near ties (margin <= {tie:.3g} nats; "
                                     "either choice is correct)"))
    selfs = [judge_self(*c, runs, ref, P, tie) for c in SELF_CHECKS]
    for sv in selfs:
        if sv.available and not sv.bitwise:
            fc = sv.first_change
            ch = (f"; {sv.generations_changed} of {len(P)} generations changed, first at prompt {fc['prompt']} "
                  f"token {fc['index']} ({'a near tie' if fc['near_tie'] else 'not a near tie'})" if fc
                  else "; no generated token changed")
            issues.append((sv.code, f"{sv.label} ({sv.baseline} vs {sv.candidate}): results differ, largest "
                                    f"logprob change {sv.max_dlogprob:.3g} nats{ch}"))
    return ServingReport(meta["engine"], results_dir, meta, tol, rms, emax, tie, FLOOR[spec.dtype], passes, selfs,
                         errors, issues)


def _f(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3g}"


def to_text(rep: ServingReport) -> str:
    m, spec = rep.meta, rep.meta["spec"]
    L = spec["prefix_len"] + spec["suffix_len"]
    lines = [f"truestar conform {rep.engine}: {rep.results_dir}",
             f"  {rep.engine} {m.get('engine_version')}, {m.get('gpu')}, {spec['dtype']}, torch {m.get('torch')}; "
             f"{spec['n_prompts']} prompts of {L} tokens ({spec['prefix_len']} shared), {spec['max_tokens']} greedy tokens",
             f"  tolerance {rep.tol:.3g} nats per position = max(floor {rep.floor:g}, {TOL_FACTOR:g} x rms "
             f"{rep.emulated_rms:.3g} of a correct CPU {spec['dtype']} run); near tie: float64 margin "
             f"<= {rep.tie:.3g} (twice the largest error of that run, {rep.emulated_max:.3g})",
             ""]
    if m.get("planted"):
        lines.append(f"  planted fault: the engine served config.json with {m['planted']} (the reference did not)")
    for s, e in rep.setting_errors.items():
        lines.append(f"  [FAILED] setting {s}: {e[:300]}")
    lines.append("  vs float64")
    hdr = f"  {'setting.pass':<30}{'positions':>10}{'over tol':>9}{'rms err':>9}{'worst err':>10}{'top1 real/tie':>15}" \
          f"{'greedy real/tie':>17}  first real greedy divergence"
    lines.append(hdr)
    for p in rep.passes:
        fr = p.first_real
        first = f"prompt {fr['prompt']} token {fr['index']}" if fr else "none"
        lines.append(f"  {p.setting + '.' + p.name:<30}{p.positions:>10}{p.over_tol:>9}"
                     f"{_f(p.rms):>9}{_f(p.worst['error'] if p.worst else None):>10}{f'{p.real_flips}/{p.tie_flips}':>15}"
                     f"{f'{p.greedy_real}/{p.greedy_tie}':>17}  {first}")
    lines += ["", "  vs itself (features meant to be lossless)",
              f"  {'check':<44}{'bitwise':>8}{'max dlogprob':>13}{'gens changed':>13}{'cached':>8}  first changed token"]
    for s in rep.selfs:
        if not s.available:
            lines.append(f"  {s.label:<44}{'n/a':>8}  {s.reason[:120] if s.reason else ''}")
            continue
        fc = s.first_change
        first = (f"prompt {fc['prompt']} token {fc['index']}{' (near tie)' if fc['near_tie'] else ''}" if fc else "none")
        cached = "-" if s.cached_tokens is None else str(s.cached_tokens)
        lines.append(f"  {s.label:<44}{'yes' if s.bitwise else 'no':>8}{_f(s.max_dlogprob):>13}"
                     f"{s.generations_changed:>13}{cached:>8}  {first}")
        for n in s.notes:
            lines.append(f"    note: {n}")
    lines.append("")
    for c, msg in rep.issues:
        lines.append(f"  {c} {msg}")
    lines.append(f"overall: {'FINDING' if rep.finding else 'OK'}")
    return "\n".join(lines)
