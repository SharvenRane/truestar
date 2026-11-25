"""`truestar port ref.py jax:cand.py` and `keras:cand.py`: a PyTorch reference against a port in
another framework, per layer (docs/PORT_ENGINES.md).

The candidate file defines `get_model(weights_path)`: it receives the reference's state dict as a
safetensors file (written by truestar.interop) and converts it the way the port's converter does.
The candidate runs in a child process (truestar.engines.xport_child), one per framework and per
Keras backend, in float64 (JAX x64, Keras floatx float64). Then, as in `truestar port`:

1. weights are matched by content (`portcheck.match_weights`: equal, transpose, permutation, ...),
   so module names never matter;
2. modules map when their matched weights cover each other;
3. every mapped candidate module is replayed alone in its framework on the reference module's
   recorded inputs (each candidate input replaced by the reference input of the same shape), so
   its error is its own, and the threshold is 10x the reference module's own float32 error;
4. the first module over threshold whose mapped modules inside all agree is named.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..capture import ROOT, Recorder, flatten_tensors
from ..codes import ERROR, Issue, severity
from ..compare import DIVERGENT, OK, SUSPECT
from ..portcheck import match_weights

F64 = torch.float64
FLOOR = 1e-9
FACTOR = 10.0
FLOAT64_NOISE = 1e-11   # float64 reorderings stay near 1e-15; above this a module computes below float64
DIVERGENT_ABOVE = 1e-6  # over threshold and above this: divergent; over threshold only: suspect


class FrameworkError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _rel(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.shape != b.shape:
        return None
    a = a.to(F64)
    b = b.to(F64)
    n = b.norm().item()
    d = (a - b).norm().item()
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


class Child:
    """A framework process speaking one JSON line per request."""

    def __init__(self, framework: str, backend: Optional[str] = None):
        env = dict(os.environ)
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        if backend:
            env["KERAS_BACKEND"] = backend
        env.setdefault("JAX_PLATFORMS", "cpu")
        env["CUDA_VISIBLE_DEVICES"] = "-1"  # a float64 CPU check; Keras on torch would otherwise place variables on the GPU
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.proc = subprocess.Popen([sys.executable, "-m", "truestar.engines.xport_child"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True,
                                     encoding="utf-8", errors="replace")
        self.framework = framework

    def request(self, req: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()[-3000:]
            raise FrameworkError("TS652", f"the {self.framework} process ended: {err.strip()}")
        rep = json.loads(line)
        if not rep.get("ok"):
            raise FrameworkError("TS652" if req["op"] == "load" and _missing(rep) else "TS021",
                                 f"{self.framework}: {rep.get('error')}\n{rep.get('trace', '')}")
        return rep

    def close(self):
        try:
            self.request({"op": "quit"}, timeout=30)
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        for s in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                s.close()
            except Exception:
                pass
        self.proc.wait(timeout=30)


def _missing(rep) -> bool:
    e = rep.get("error", "")
    return "ModuleNotFoundError" in e or "No module named" in e or "ImportError" in e


@dataclass
class Pair:
    reference: str
    candidate: str
    ref_type: str
    cand_type: str
    call: int = 0
    rel_err: Optional[float] = None
    threshold: float = FLOOR
    accumulated: Optional[float] = None  # candidate's own full run output against the reference's
    status: str = OK
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"reference": self.reference, "candidate": self.candidate, "reference_type": self.ref_type,
                "candidate_type": self.cand_type, "call": self.call, "rel_err": self.rel_err,
                "accumulated_rel_err": self.accumulated, "threshold": self.threshold, "status": self.status,
                "note": self.note or None}


@dataclass
class BackendReport:
    label: str
    version: str = ""
    status: str = OK
    output_rel_err: Optional[float] = None
    pairs: List[Pair] = field(default_factory=list)
    blamed: Optional[Pair] = None
    edge_kinds: Dict[str, int] = field(default_factory=dict)
    ref_unmatched: List[str] = field(default_factory=list)
    cand_unmatched: List[str] = field(default_factory=list)
    ref_unmapped: List[str] = field(default_factory=list)
    lower_precision: List[str] = field(default_factory=list)
    hint: str = ""
    issues: List[Issue] = field(default_factory=list)
    seconds: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.label, "version": self.version, "status": self.status,
                "output_rel_err": self.output_rel_err, "blamed": self.blamed.as_dict() if self.blamed else None,
                "hint": self.hint or None, "pairs_checked": len(self.pairs),
                "pairs_over_threshold": [p.as_dict() for p in self.pairs if p.status != OK][:5],
                "weights": {"kinds": self.edge_kinds, "reference_unmatched": self.ref_unmatched,
                            "candidate_unmatched": self.cand_unmatched},
                "reference_unmapped": self.ref_unmapped[:20], "lower_precision_modules": self.lower_precision[:20], "codes": [i.code for i in self.issues],
                "seconds": self.seconds}


@dataclass
class FrameworkPortReport:
    reference_name: str
    candidate_name: str
    kind: str
    backends: List[BackendReport] = field(default_factory=list)
    status: str = OK
    issues: List[Issue] = field(default_factory=list)
    explanation: str = ""
    seconds: float = 0.0

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for i in self.issues:
            if i.code not in out:
                out.append(i.code)
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "codes": self.codes,
                "issues": [i.as_dict() for i in self.issues], "backends": [b.as_dict() for b in self.backends],
                "explanation": self.explanation, "seconds": self.seconds}

    def to_text(self) -> str:
        L = [f"truestar port: reference {self.reference_name} vs {self.kind} candidate {self.candidate_name}"]
        for b in self.backends:
            k = ", ".join(f"{n} {k}" for k, n in sorted(b.edge_kinds.items()))
            out = "n/a" if b.output_rel_err is None else f"{b.output_rel_err:.2e}"
            L.append(f"[{b.status.upper()}] {b.label} ({b.version}): weights {k}; {len(b.pairs)} module pairs "
                     f"replayed; final output relative error {out}")
            if b.blamed:
                p = b.blamed
                L.append(f"  first module over threshold: candidate {p.candidate} ({p.cand_type}) vs reference "
                         f"{p.reference} ({p.ref_type}), own error {p.rel_err:.2e}, threshold {p.threshold:.1e}")
                if b.hint:
                    L.append(f"  {b.hint}")
        L.append(f"status: {self.status.upper()}")
        if self.explanation:
            L += ["", self.explanation]
        return "\n".join(L)


# ----------------------------------------------------------------------------- reference side


def _closures(names: Sequence[str], params: Sequence[str], sep: str = ".") -> Dict[str, set]:
    out = {}
    for m in names:
        pre = (m + sep) if m else ""
        out[m] = {p for p in params if p.startswith(pre)}
    return out


def _thr(module: torch.nn.Module, call, out64: torch.Tensor) -> float:
    try:
        m32 = copy.deepcopy(module).float()
        args = [a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in call.args]
        with torch.no_grad():
            o = flatten_tensors(m32(*args, **call.kwargs))[0]
        e = _rel(o, out64)
        return max(FACTOR * (e or 0.0), FLOOR)
    except Exception:
        return FLOOR * 1e3


def framework_port_check(reference: torch.nn.Module, inputs: Any, kind: str, candidate_file: str,
                         backends: Optional[Sequence[str]] = None, reference_name: str = "reference",
                         workdir: Optional[str] = None) -> FrameworkPortReport:
    t_all = time.perf_counter()
    if kind not in ("jax", "keras"):
        raise FrameworkError("TS652", f"unknown framework {kind!r} (jax or keras)")
    rep = FrameworkPortReport(reference_name, os.path.basename(candidate_file), kind)
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    ref = copy.deepcopy(reference).to(F64).eval()
    with torch.no_grad(), Recorder(ref, keep_inputs=True) as rec:
        ref_out = flatten_tensors(ref(*args))[0]
    calls = rec.calls
    with torch.no_grad():
        r32 = copy.deepcopy(reference).float().eval()
        ref32_err = _rel(flatten_tensors(r32(*[a.float() if a.is_floating_point() else a for a in args]))[0], ref_out)
    ref_params = {n: p.detach().to(F64) for n, p in ref.named_parameters()}
    mods = dict(ref.named_modules())
    tmp = tempfile.mkdtemp(prefix="truestar_xport_", dir=workdir)
    from ..interop import save_file

    wfile = os.path.join(tmp, "reference.safetensors")
    save_file({k: v.detach().contiguous() for k, v in reference.state_dict().items()}, wfile)
    ifile = os.path.join(tmp, "inputs.safetensors")
    save_file({f"in{i}": a.contiguous() for i, a in enumerate(args)}, ifile)
    labels = [(kind, None)] if kind == "jax" else [(f"keras:{b}", b) for b in (backends or ["jax", "torch"])]
    for label, backend in labels:
        rep.backends.append(_one_backend(label, kind, backend, candidate_file, wfile, ifile, tmp, ref, mods, calls,
                                         ref_params, ref_out, ref32_err))
    for b in rep.backends:
        rep.issues.extend(b.issues)
    sts = [b.status for b in rep.backends]
    rep.status = DIVERGENT if DIVERGENT in sts else SUSPECT if SUSPECT in sts else OK
    if len(rep.backends) > 1 and OK in sts and DIVERGENT in sts:
        bad = [b.label for b in rep.backends if b.status == DIVERGENT]
        good = [b.label for b in rep.backends if b.status == OK]
        rep.issues.append(Issue("TS656", f"{', '.join(bad)} differ from the reference while {', '.join(good)} match"))
    rep.explanation = _explain(rep)
    rep.seconds = time.perf_counter() - t_all
    return rep


def _one_backend(label, kind, backend, candidate_file, wfile, ifile, tmp, ref, mods, calls, ref_params, ref_out,
                 ref32_err=None):
    br = BackendReport(label)
    t0 = time.perf_counter()
    child = Child(label, backend)
    from ..interop import load_file, save_file

    try:
        load = child.request({"op": "load", "framework": kind, "file": os.path.abspath(candidate_file),
                              "weights": wfile, "inputs": ifile, "out": os.path.join(tmp, label.replace(":", "_"))})
        br.seconds["load_and_run"] = time.perf_counter() - t0
        br.version = load["version"]
        cand_params = {k: v.to(F64) for k, v in load_file(load["params"], mmap=False).items()}
        recs = load_file(load["records"], mmap=False)
        final = recs[load["final"][0]] if load["final"] else None
        br.output_rel_err = _rel(final, ref_out) if final is not None else None
        wm = match_weights(ref_params, cand_params, {n: i for i, n in enumerate(ref_params)},
                           {n: i for i, n in enumerate(cand_params)})
        br.edge_kinds = wm.counts()
        br.ref_unmatched = list(wm.ref_unmatched)
        br.cand_unmatched = list(wm.cand_unmatched)
        if br.ref_unmatched or br.cand_unmatched:
            br.issues.append(Issue("TS655", f"{label}: reference weights without a counterpart: "
                                            f"{br.ref_unmatched[:6]}; candidate weights matching nothing: "
                                            f"{br.cand_unmatched[:6]}", label))
        r2c = {}
        for e in wm.edges:
            r2c.setdefault(e.ref, set()).add(e.cand)
        c2r = {}
        for e in wm.edges:
            c2r.setdefault(e.cand, set()).add(e.ref)
        rclos = _closures(list(mods), list(ref_params))
        cmods = [m["path"] for m in load["modules"]]
        ctype = {m["path"]: m["type"] for m in load["modules"]}
        cown = {m["path"]: m["params"] for m in load["modules"]}
        cclos = {p: {x for q in cmods if q == p or p == "" or q.startswith(p + ".") for x in cown[q]} for p in cmods}
        pairs: List[Tuple[str, str]] = []
        used = set()
        for rm, rc in rclos.items():
            rc = {x for x in rc if x in r2c}
            if not rc:
                continue
            image = set().union(*(r2c[x] for x in rc))
            pre = set().union(*(c2r.get(x, set()) for x in image))
            if pre != rc:
                continue
            cands = [p for p in cmods if cclos[p] == image and p not in used]
            if cands:
                cm = min(cands, key=lambda p: (p.count("."), cmods.index(p)))
                used.add(cm)
                pairs.append((rm, cm))
        mapped_ref = {r for r, _ in pairs}
        br.ref_unmapped = [m for m in mods if rclos[m] and m not in mapped_ref]
        # calls
        rcalls: Dict[str, List[Any]] = {}
        for c in sorted(calls, key=lambda c: c.start):
            rcalls.setdefault("" if c.name == ROOT else c.name, []).append(c)
        ccalls: Dict[str, List[Dict]] = {}
        for i, c in enumerate(load["calls"]):
            c = dict(c, n=i)
            ccalls.setdefault(c["module"], []).append(c)
        jobs, meta = [], []
        arrays: Dict[str, torch.Tensor] = {}
        for rm, cm in pairs:
            for k, (rc, cc) in enumerate(zip(rcalls.get(rm, []), ccalls.get(cm, []))):
                rin = [a for a in flatten_tensors((rc.args, rc.kwargs))]
                jargs = []
                for j, a in enumerate(cc["args"]):
                    if isinstance(a, str):
                        t = recs[a]
                        pick = _substitute(rin, t)
                        key = f"p{len(meta)}a{j}"
                        arrays[key] = pick.contiguous()
                        jargs.append(key)
                    else:
                        jargs.append(a)
                jobs.append({"module": cm, "args": jargs})
                cout = recs[cc["outputs"][0]] if cc["outputs"] else None
                meta.append((rm, cm, k, rc, cout))
        afile = os.path.join(tmp, f"{label.replace(':', '_')}_args.safetensors")
        if arrays:
            save_file(arrays, afile)
        for j in jobs:
            j["file"] = afile
        ofile = os.path.join(tmp, f"{label.replace(':', '_')}_out.safetensors")
        t1 = time.perf_counter()
        child.request({"op": "replay", "jobs": jobs, "out": ofile})
        br.seconds["replay"] = time.perf_counter() - t1
        outs = load_file(ofile, mmap=False) if jobs else {}
        lower: List[str] = []
        inside: List[str] = []
        for j, (rm, cm, k, rc, cout) in enumerate(meta):
            r64 = rc.outputs[0] if rc.outputs else None
            o = outs.get(f"j{j}.out0")
            p = Pair(rm or ROOT, cm or "<model>", type(mods[rm]).__name__, ctype[cm], k)
            if o is not None and o.is_floating_point() and o.dtype != F64:
                lower.append(cm or "<model>")
            if r64 is None or o is None:
                p.note = "no output to compare"
            else:
                p.rel_err = _rel(o, r64)
                if p.rel_err is None:
                    p.note = f"output shapes differ: {tuple(o.shape)} vs {tuple(r64.shape)}"
                p.threshold = _thr(mods[rm], rc, r64)
                p.accumulated = _rel(cout, r64) if cout is not None else None
                if p.rel_err is not None and p.rel_err > p.threshold:
                    p.status = DIVERGENT if p.rel_err > DIVERGENT_ABOVE else SUSPECT
            if (p.rel_err is not None and p.rel_err > FLOAT64_NOISE and o is not None and o.dtype == F64
                    and (cm or "<model>") not in lower):
                inside.append(cm or "<model>")  # float64 out, float32 level error: float32 inside
            br.pairs.append(p)
        # the same rule as for each module: under 10x the reference's own float32 noise is the same model
        final_thr = max(FLOOR, FACTOR * (ref32_err or 0.0))
        if lower or inside or (final is not None and final.dtype != F64):
            br.lower_precision = lower + inside
            what = []
            if lower:
                what.append(f"{len(lower)} return float32 (for example {', '.join(lower[:3])})")
            if inside:
                what.append(f"{len(inside)} return float64 but differ at float32 level, so they compute in float32 "
                            f"inside (for example {', '.join(inside[:3])})")
            br.issues.append(Issue("TS658", f"{label}: candidate modules below float64: " + "; ".join(what)
                                   + "; the verdict uses float32 noise thresholds", label))
        # blame: over threshold and every mapped module inside agrees; first in call order
        bad = [p for p in br.pairs if p.status != OK]

        def within(a: Pair, b: Pair) -> bool:
            return b.reference != a.reference and (a.reference == ROOT or b.reference.startswith(a.reference + "."))
        own = [p for p in bad if not any(within(p, q) for q in bad)]
        if own:
            br.blamed = own[0]
            br.status = own[0].status
            br.hint = _hint(br.blamed, wm, ref_params)
            code = "TS653" if br.status == DIVERGENT else "TS654"
            br.issues.append(Issue(code, f"{label}: candidate {br.blamed.candidate} ({br.blamed.cand_type}) computes "
                                         f"differently from reference {br.blamed.reference} ({br.blamed.ref_type}) on the "
                                         f"same inputs, own error {br.blamed.rel_err:.2e}, threshold {br.blamed.threshold:.1e}",
                                   label))
        elif br.ref_unmatched or br.cand_unmatched:
            br.status = DIVERGENT
        elif br.output_rel_err is not None and br.output_rel_err > final_thr:
            br.status = DIVERGENT
            br.issues.append(Issue("TS653", f"{label}: final output differs ({br.output_rel_err:.2e}) but no mapped module "
                                            "is to blame", label))
    finally:
        child.close()
    br.seconds["total"] = time.perf_counter() - t0
    return br


def _substitute(ref_inputs: List[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
    """The reference input of the same shape and dtype closest in value to the candidate's own input
    `t` (equal when everything upstream agrees), else `t` itself."""
    same = [r for r in ref_inputs if r.shape == t.shape and r.dtype == t.dtype]
    return min(same, key=lambda r: float((r.double() - t.double()).norm())) if same else t


def _hint(p: Pair, wm, ref_params) -> str:
    """A 2 D weight inside the blamed module stored in another orientation than the port's other
    2 D weights: the converter likely missed (or added) a transpose."""
    kinds2d = [e.kind for e in wm.edges if ref_params[e.ref].ndim == 2]
    if not kinds2d:
        return ""
    common = Counter(kinds2d).most_common(1)[0][0]
    pre = "" if p.reference == ROOT else p.reference + "."
    odd = [e for e in wm.edges if e.ref.startswith(pre) and ref_params[e.ref].ndim == 2 and e.kind != common]
    if odd:
        e = odd[0]
        return (f"weight {e.cand} holds {e.ref} as '{e.kind}' while the port's other 2 D weights are "
                f"'{common}': the converter likely missed a transpose there")
    return ""


def _explain(rep: FrameworkPortReport) -> str:
    parts = []
    for b in rep.backends:
        if b.blamed:
            p = b.blamed
            s = (f"{b.label}: {p.candidate} ({p.cand_type}) computes differently from {p.reference} ({p.ref_type}) "
                 f"given the reference's inputs (own error {p.rel_err:.2e}, threshold {p.threshold:.1e}); every "
                 f"mapped module inside it agrees, so the difference is in its own code.")
            if b.hint:
                s += " " + b.hint[0].upper() + b.hint[1:] + "."
            parts.append(s)
        elif b.status != OK:
            parts.append(f"{b.label}: weights do not line up ({len(b.ref_unmatched)} reference, "
                         f"{len(b.cand_unmatched)} candidate unmatched).")
    if "TS656" in rep.codes:
        parts.append("The same candidate code gives another answer on one backend only.")
    if not parts:
        low = [b.label for b in rep.backends if b.lower_precision]
        parts.append("Same model on " + ", ".join(b.label for b in rep.backends) + ": every mapped module agrees "
                     "with the reference within " + ("float32 noise (" + ", ".join(low) + " compute parts in float32, "
                     "TS658)" if low else "float64 noise") + ".")
    return " ".join(parts)


def report_to_agent(rep: FrameworkPortReport, reference_file: Optional[str], candidate_file: str) -> Dict[str, Any]:
    from ..agent import _envelope, exit_code_for

    doc = _envelope(rep.status, exit_code_for(rep.status))
    doc["command"] = "port"
    doc["reference"] = {"name": rep.reference_name, "file": os.path.abspath(reference_file) if reference_file else None}
    doc["candidate"] = {"name": rep.candidate_name, "file": os.path.abspath(candidate_file), "kind": rep.kind}
    doc["precision"] = "float64 in every framework"
    doc.update(rep.as_dict())
    acts = []
    for b in rep.backends:
        if b.blamed:
            acts.append({"kind": "inspect_candidate_module", "command": None, "backend": b.label,
                         "module": b.blamed.candidate,
                         "description": f"Read {b.blamed.candidate} ({b.blamed.cand_type}) next to the reference "
                                        f"{b.blamed.reference}. {b.hint}".strip()})
    if acts and reference_file:
        acts.append({"kind": "rerun", "description": "Rerun after a fix; exit 0 means the port computes the same model.",
                     "command": f'truestar port "{os.path.abspath(reference_file)}" "{rep.kind}:{os.path.abspath(candidate_file)}" --format agent'})
    doc["next_actions"] = acts
    return doc
