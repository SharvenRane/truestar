"""`parallel_check`: one training step under a layout against the single device program, with blame.

Both arms run in float64. Under float64 a correct layout differs from the single device run only
by reduction order, about 1e-16 relative, so one fixed threshold (1e-9) separates a layout bug from
noise with seven orders of margin and needs no calibration run. The parallel arm's per rank
pieces (module outputs, their gradients, parameter gradients after every sync, updated weights)
are reassembled into whole tensors and compared one by one in this order, so the first difference
names its cause:

1. shards: every rank's initial piece of every split weight against the plan's split of the
   reference weight (a wrong split is named with the split it actually is);
2. forward: the first module, in the order forwards finish, whose output differs;
3. loss: the loss value, when every output matches (normalization), with the normalization the
   layout actually computed found among the usual wrong ones;
4. backward: the last module whose output gradient differs while every later one matches; the
   module after it produced that gradient (a backward collective);
5. parameter gradients: rank disagreement on a replicated weight (a missing sync, and whether the
   ranks' sum is the reference), one factor over every gradient (a scaling collective or the
   accumulation), else the module whose own gradients differ;
6. update: right gradients, wrong weights after the optimizer (clipping on a local norm);
7. drift: replicated weights that differ between ranks after the step.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..codes import Issue
from .comm import ThreadComm, _Hub
from .layout import Layout, LayoutError, Plan, Style, auto_plan, is_sharded, parse_layout, shard, unshard, validate_plan
from .runtime import FAULTS, MSE, RankRecord, run_rank, run_reference

TOL = 1e-9


def _rel(a: Optional[torch.Tensor], b: Optional[torch.Tensor], floor: float = 0.0) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None or a.shape != b.shape:
        return math.inf
    a, b = a.double(), b.double()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        return 0.0 if torch.equal(a, b) else math.inf
    if a.numel() == 0:
        return 0.0
    d = float((a - b).abs().max())
    s = max(float(b.abs().max()), floor)
    return d / s if s > 0 else d


@dataclasses.dataclass
class Finding:
    code: str
    message: str
    module: Optional[str] = None
    collective: Optional[str] = None
    error: Optional[float] = None

    def as_dict(self) -> dict:
        d = Issue(self.code, self.message).as_dict()
        d.pop("backend", None)
        d.update({"module": self.module, "collective": self.collective, "error": self.error})
        return d


@dataclasses.dataclass
class ParallelReport:
    layout: Layout
    plan: Dict[str, str]
    plan_source: str
    ranks: str
    findings: List[Finding]
    loss_reference: float
    loss_parallel: float
    compared: Dict[str, int]
    worst: Dict[str, float]
    timings: Dict[str, float]
    collectives: int
    faults: List[str]
    model_name: str = "model"
    sources: Dict[str, str] = dataclasses.field(default_factory=dict)
    devices: List[str] = dataclasses.field(default_factory=list)  # real ranks: one line per rank

    @property
    def real(self) -> bool:
        return self.ranks == "devices"

    @property
    def status(self) -> str:
        return "divergent" if any(f.code not in ("TS690", "TS820", "TS825") for f in self.findings) else "ok"

    @property
    def codes(self) -> List[str]:
        return sorted({f.code for f in self.findings})

    def first(self) -> Optional[Finding]:
        return next((f for f in self.findings if f.code not in ("TS690", "TS820", "TS825")), None)

    def to_text(self) -> str:
        kind = "real ranks" if self.real else "simulated ranks"
        lines = [f"truestar parallel: {self.model_name}, layout {self.layout.text()}, "
                 f"{self.layout.world} {kind} ({self.ranks}), one step, float64 both arms"]
        if self.faults:
            lines.append(f"  injected faults: {', '.join(self.faults)}")
        plan = ", ".join(f"{k}={v}" for k, v in self.plan.items()) or "(no tensor parallel split)"
        lines.append(f"  plan ({self.plan_source}): {plan}")
        lines.append(f"  loss: single device {self.loss_reference:.10g}, layout {self.loss_parallel:.10g}")
        lines.append("  compared: " + ", ".join(f"{v} {k}" for k, v in self.compared.items()))
        lines.append("  worst error: " + ", ".join(f"{k} {v:.1e}" for k, v in self.worst.items())
                     + f" (threshold {TOL:.0e})")
        real = [f for f in self.findings if f.code not in ("TS690", "TS820", "TS825")]
        if not real:
            lines.append("overall: OK (the layout computes the single device step)")
        else:
            lines.append(f"overall: DIVERGENT ({len(real)} finding{'s' if len(real) > 1 else ''})")
            for f in real:
                where = f" [{f.module}]" if f.module else ""
                lines.append(f"  {f.code}{where} {f.message}")
                if f.module and f.module in self.sources:
                    lines.append(f"      defined at {self.sources[f.module]}")
                if f.collective:
                    lines.append(f"      collective: {f.collective}")
        lines.append(f"  {self.collectives} collectives on rank 0; reference {self.timings['reference']:.2f} s, "
                     f"layout {self.timings['parallel']:.2f} s, diagnosis {self.timings['diagnosis']:.2f} s")
        if self.real:
            lines.append("  note TS820: " + "; ".join(self.devices))
            lines += [f"  warning TS825: {f.message}" for f in self.findings if f.code == "TS825"]
        else:
            lines.append("  note TS690: ranks are simulated on this CPU (threads or gloo processes); the "
                         "arithmetic of the layout is real, no GPU interconnect is involved")
        return "\n".join(lines)

    def to_agent(self, model_file: Optional[str] = None) -> dict:
        from ..agent import _envelope

        exit_code = 1 if self.status == "divergent" else 0
        doc = _envelope(self.status, exit_code)
        first = self.first()
        doc.update({
            "command": "parallel",
            "model_file": os.path.abspath(model_file) if model_file else None,
            "layout": dataclasses.asdict(self.layout),
            "plan": self.plan, "plan_source": self.plan_source, "ranks": self.ranks,
            "simulated": not self.real,
            "devices": self.devices,
            "loss": {"reference": self.loss_reference, "parallel": self.loss_parallel},
            "compared": self.compared,
            "worst_error": {k: (v if math.isfinite(v) else None) for k, v in self.worst.items()},
            "threshold": TOL,
            "blamed_module": first.module if first else None,
            "blamed_collective": first.collective if first else None,
            "source": self.sources.get(first.module) if first and first.module else None,
            "codes": self.codes,
            "issues": [dict(f.as_dict(), error=(f.error if f.error is None or math.isfinite(f.error) else None))
                       for f in self.findings],
            "injected_faults": self.faults,
            "timings_s": self.timings,
        })
        actions = []
        if first is not None:
            actions.append({"kind": "fix_layout", "description": f"{first.code}: {first.message}",
                            "command": None})
            if model_file:
                actions.append({"kind": "rerun", "description": "Rerun the check after the fix.",
                                "command": f'truestar parallel "{os.path.abspath(model_file)}" --layout '
                                           f'{self.layout.text()}' + ('' if self.ranks == 'threads' else f' --ranks {self.ranks}')
                                           + ' --format agent'})
        doc["next_actions"] = actions
        return doc


# ----------------------------------------------------------------------------- running both arms


def _to64(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.detach().double()
    return x


def _run_threads(model, inputs, targets, loss, layout, plan, optimizer, lr, clip, faults, timeout):
    hub = _Hub(timeout)
    recs: List[Optional[RankRecord]] = [None] * layout.world

    def body(r):
        m = copy.deepcopy(model)
        comm = ThreadComm(hub, r, layout.world)
        recs[r] = run_rank(comm, m, inputs, targets, loss, layout, plan, optimizer, lr, clip, faults, hub.abort)

    threads = [threading.Thread(target=body, args=(r,), name=f"truestar-rank{r}", daemon=True)
               for r in range(layout.world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 30)
    for r, t in enumerate(threads):
        if t.is_alive():
            hub.abort()
            t.join(10)
        if recs[r] is None:
            rr = RankRecord(r, layout.tp_rank(r), layout.dp_rank(r))
            rr.error = "the rank did not finish (hang past the timeout)"
            recs[r] = rr
    return recs


def _run_gloo(model_file, state, inputs, targets, loss, layout, plan, optimizer, lr, clip, faults, timeout):
    import torch.multiprocessing as mp

    from . import worker

    with tempfile.TemporaryDirectory(prefix="truestar_parallel_") as d:
        payload = os.path.join(d, "payload.pt")
        torch.save({"state": state, "inputs": inputs, "targets": targets, "loss": loss, "layout": layout,
                    "plan": plan, "optimizer": optimizer, "lr": lr, "clip": clip, "faults": list(faults),
                    "model_file": os.path.abspath(model_file)}, payload)
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker.gloo_rank, args=(r, layout.world, d, timeout)) for r in range(layout.world)]
        for p in procs:
            p.start()
        t0 = time.time()
        failed_at = None
        try:
            while any(p.is_alive() for p in procs) and time.time() - t0 < timeout + 60:
                if failed_at is None and any(p.exitcode == 3 for p in procs):
                    failed_at = time.time()  # a rank failed: its peers wait in a collective that never completes
                if failed_at is not None and time.time() - failed_at > 3.0:
                    break
                time.sleep(0.05)
        finally:
            for p in procs:
                if p.is_alive():
                    p.kill()
                p.join(timeout=10)
        recs = []
        for r in range(layout.world):
            f = os.path.join(d, f"rank{r}.pt")
            if os.path.exists(f):
                recs.append(torch.load(f, weights_only=False))
            elif failed_at is not None:
                rr = RankRecord(r, layout.tp_rank(r), layout.dp_rank(r))
                rr.error = "aborted: another rank failed, so this rank's collective never completed"
                recs.append(rr)
            else:
                rr = RankRecord(r, layout.tp_rank(r), layout.dp_rank(r))
                rr.error = f"the gloo process exited with code {procs[r].exitcode} and wrote no result"
                recs.append(rr)
        return recs


# ----------------------------------------------------------------------------- reassembly


def _join_tp(pieces: List[torch.Tensor], hint, want_tail: Tuple[int, ...], nd: int):
    """Join one (dp, micro) slot's tp pieces. Returns (tensor, tp disagreement)."""
    tries = [hint] + [("rep",)] + [("split", k) for k in range(1, nd)]
    for h in tries:
        if h[0] == "rep":
            if all(p.shape == pieces[0].shape for p in pieces) and tuple(pieces[0].shape[1:]) == want_tail:
                dis = max((_rel(p, pieces[0]) for p in pieces[1:]), default=0.0)
                return pieces[0], dis
        else:
            k = h[1] % nd
            parts = h[2] if len(h) > 2 else 1
            try:
                if parts > 1:
                    per = pieces[0].shape[k] // parts
                    j = torch.cat([p.narrow(k, q * per, per) for q in range(parts) for p in pieces], dim=k)
                else:
                    j = torch.cat(pieces, dim=k)
            except RuntimeError:
                continue
            if tuple(j.shape[1:]) == want_tail:
                return j, 0.0
    return None, 0.0


def _assemble(slots, layout: Layout, ref: torch.Tensor, hint):
    """slots[(d, m)][j]: piece of dp rank d, micro batch m, tp rank j. Returns (whole, tp disagreement)."""
    if ref.dim() == 0:
        return None, 0.0
    tail = tuple(ref.shape[1:])
    parts, dis = [], 0.0
    for d in range(layout.dp):
        for m in range(layout.ga):
            pieces = slots[(d, m)]
            if any(p is None for p in pieces):
                return None, 0.0
            j, di = _join_tp(pieces, hint, tail, ref.dim())
            if j is None:
                return None, 0.0
            parts.append(j)
            dis = max(dis, di)
    whole = torch.cat(parts, dim=0)
    return (whole if whole.shape == ref.shape else None), dis


def _module_slots(recs, layout, name, c, k, ncalls, field):
    slots = {}
    for d in range(layout.dp):
        for m in range(layout.ga):
            row = []
            for j in range(layout.tp):
                calls = recs[d * layout.tp + j].calls.get(name, [])
                idx = m * ncalls + c
                if idx >= len(calls) or k >= len(getattr(calls[idx], field)):
                    row.append(None)
                else:
                    row.append(getattr(calls[idx], field)[k])
            slots[(d, m)] = row
    return slots


def _hint(recs, name, c):
    calls = recs[0].calls.get(name, [])
    return calls[c].hint if c < len(calls) else ("rep",)


def _param_whole(recs, layout, plan, pname, field):
    """Whole tensor per dp replica of a parameter field (init, grads, updated) and the tp disagreement."""
    mname, _, short = recs[0].owner[pname]
    st = plan.get(mname)
    wholes, dis = [], 0.0
    for d in range(layout.dp):
        pieces = [getattr(recs[d * layout.tp + j], field).get(pname) for j in range(layout.tp)]
        if any(p is None for p in pieces):
            wholes.append(None)
            continue
        if is_sharded(st, short):
            try:
                wholes.append(unshard(pieces, st, short))
            except RuntimeError:
                wholes.append(None)
        else:
            dis = max(dis, max((_rel(p, pieces[0]) for p in pieces[1:]), default=0.0))
            wholes.append(pieces[0])
    return wholes, dis


def _describe_shard(local: torch.Tensor, full: torch.Tensor, tp: int, rank: int) -> str:
    cands = []
    for parts in (1, 2, 3, 4):
        if full.shape[0] % (parts * tp) == 0:
            cands.append((f"the split along dim 0 into {parts} part{'s' if parts > 1 else ''} per rank" if parts > 1
                          else "a split along dim 0 as one block", shard(full, Style("column", parts), "weight", tp, rank)))
    if full.dim() == 2:
        for dim in (0, 1):
            if full.shape[dim] % tp == 0:
                cands.append((f"chunk {rank} of dim {dim}", full.chunk(tp, dim=dim)[rank]))
    for text, t in cands:
        if t.shape == local.shape and torch.equal(t, local):
            return text
    return f"no split of the reference weight (shape {tuple(local.shape)} of {tuple(full.shape)})"


_FACTORS = [("dp", lambda L: L.dp), ("1/dp", lambda L: 1 / L.dp), ("ga", lambda L: L.ga), ("1/ga", lambda L: 1 / L.ga),
            ("tp", lambda L: L.tp), ("1/tp", lambda L: 1 / L.tp), ("world", lambda L: L.world),
            ("1/world", lambda L: 1 / L.world)]

_FACTOR_CAUSE = {
    "dp": ("gradient bucket all reduce over dp",
           "the all reduce sums gradients of per replica means: divide every loss by the global count "
           "(all reduce the count) or average the sum"),
    "1/dp": ("gradient bucket all reduce over dp",
             "gradients are averaged over dp although every loss is already divided by the global count"),
    "1/ga": ("gradient accumulation",
             "accumulated gradients are divided by ga although every micro loss already carries 1/ga "
             "(divided twice)"),
    "ga": ("gradient accumulation", "micro batch losses are not divided by ga before backward"),
    "tp": ("tensor parallel gradient sync", "gradients are summed over tp ranks that already hold the full gradient"),
    "1/tp": ("tensor parallel gradient sync", "gradients are divided by tp"),
    "world": ("gradient bucket all reduce", "gradients are summed over every rank instead of over dp"),
    "1/world": ("gradient bucket all reduce", "gradients are divided by the world size instead of dp"),
}


def _match_factor(k: float, layout: Layout) -> List[str]:
    return [name for name, f in _FACTORS if f(layout) != 1 and abs(k - f(layout)) <= 1e-6 * abs(f(layout))]


def _loss_hypotheses(out: torch.Tensor, tgt, loss, layout: Layout) -> List[Tuple[str, float]]:
    """Loss values the usual wrong normalizations give on the same outputs and targets."""
    hyp = []
    outs = out.chunk(layout.dp * layout.ga, dim=0)
    tgts = tgt.chunk(layout.dp * layout.ga, dim=0)
    micro = [float(loss.sum(o, t) / loss.count(t)) for o, t in zip(outs, tgts)]
    counts = [float(loss.count(t)) for t in tgts]
    hyp.append(("the mean of per micro batch means (micro batch counts "
                f"{', '.join(f'{c:g}' for c in counts)}): divide every micro batch's loss sum by the global "
                "count", sum(micro) / len(micro)))
    if layout.dp > 1:
        reps = []
        for d in range(layout.dp):
            o = torch.cat(outs[d * layout.ga:(d + 1) * layout.ga])
            t = torch.cat(tgts[d * layout.ga:(d + 1) * layout.ga])
            reps.append(float(loss.sum(o, t) / loss.count(t)))
        hyp.append(("the sum of per replica means: divide by the global count, not the replica's", sum(reps)))
        hyp.append(("the mean of per replica means: divide by the global count, not the replica's",
                    sum(reps) / len(reps)))
    return hyp


def _source(m: nn.Module) -> Optional[str]:
    try:
        f = inspect.getsourcefile(type(m))
        _, line = inspect.getsourcelines(type(m))
        return f"{f}:{line} ({type(m).__name__})"
    except (TypeError, OSError):
        return None


def _diagnose(ref: RankRecord, recs: List[RankRecord], layout: Layout, plan: Plan, model: nn.Module,
              out_ref, targets, loss) -> Tuple[List[Finding], Dict[str, int], Dict[str, float]]:
    F: List[Finding] = []
    compared = {"module outputs": 0, "output gradients": 0, "parameter gradients": 0, "updated weights": 0}
    worst = {"forward": 0.0, "backward": 0.0, "gradients": 0.0, "update": 0.0}
    mods = dict(model.named_modules())
    sty = lambda n: plan[n].text() if n in plan else "replicated"  # noqa: E731

    failed = [r for r in recs if r.error]
    if failed:
        own = [r for r in failed if not r.error.startswith("aborted")] or failed
        for r in own:
            msg = f"rank {r.rank} (tp {r.tp_rank}, dp {r.dp_rank}) failed under the layout: {r.error}"
            if r.error_module:
                msg += f"; inside {r.error_module} ({type(mods[r.error_module]).__name__}, {sty(r.error_module)})" \
                    if r.error_module in mods else f"; inside {r.error_module}"
            if r.error_where:
                msg += f" at {r.error_where}"
            F.append(Finding("TS689", msg, module=r.error_module if r.error_module in mods else None))
        return F, compared, worst

    # 1. shards
    for pname, (mname, st_text, short) in recs[0].owner.items():
        st = plan.get(mname)
        if not is_sharded(st, short):
            continue
        for r in recs:
            want = shard(ref.init[pname], st, short, layout.tp, r.tp_rank)
            got = r.init[pname]
            if got.shape != want.shape or not torch.equal(got, want):
                F.append(Finding("TS681", f"{pname} on tp rank {r.tp_rank} is not the plan's shard ({st.text()}): "
                                          f"it is {_describe_shard(got, ref.init[pname], layout.tp, r.tp_rank)}. "
                                          "Every output of the module is wrong from here on.",
                                 module=mname, error=_rel(got, want) if got.shape == want.shape else math.inf))
                break
    # 2. forward, in the order forwards finish
    first_bad = None
    seen = {}
    leaf_out_ok: Dict[Tuple[str, int], bool] = {}
    for name in ref.order:
        c = seen.get(name, 0)
        seen[name] = c + 1
        ncalls = len(ref.calls[name])
        for k, rt in enumerate(ref.calls[name][c].out):
            whole, dis = _assemble(_module_slots(recs, layout, name, c, k, ncalls, "out"), layout, rt, _hint(recs, name, c))
            if whole is None:
                continue
            compared["module outputs"] += 1
            e = max(_rel(whole, rt), dis)
            worst["forward"] = max(worst["forward"], e)
            if e > TOL and first_bad is None:
                first_bad = (name, c, k, e, whole, rt, dis)
    if first_bad is not None and not any(f.code == "TS681" for f in F):
        name, c, k, e, whole, rt, dis = first_bad
        m = mods.get("" if name == "<model>" else name)
        msg = (f"output under the layout differs from the single device run: error {e:.1e}; every module "
               f"that finished before it matches")
        if dis > TOL:
            msg += f"; tp ranks disagree on it by {dis:.1e}"
        if isinstance(m, nn.Linear) and name in plan and plan[name].kind == "row" and m.bias is not None:
            ref_bias = ref.init[f"{name}.bias"]
            if _rel(whole - rt, (layout.tp - 1) * ref_bias.expand_as(rt)) < 1e-6:
                msg += (f". It is exactly {layout.tp - 1} extra copies of the bias: the bias is added on every tp "
                        "rank before the sum; add it once, after the all reduce")
        coll = f"{name} output all reduce" if name in plan and plan[name].kind == "row" else None
        F.append(Finding("TS681", f"{name} ({type(m).__name__ if m is not None else '?'}, {sty(name)}) {msg}",
                         module=None if name == "<model>" else name, collective=coll, error=e))
    if F:
        return F, compared, worst
    # 3. loss
    lp = recs[0].loss
    le = abs(lp - ref.loss) / max(abs(ref.loss), 1e-300)
    if le > TOL:
        msg = (f"loss {lp:.10g} under the layout, {ref.loss:.10g} single device (ratio {lp / ref.loss:.6f}) "
               "while every module output matches: the loss is normalized differently")
        for text, v in _loss_hypotheses(out_ref, targets, loss, layout):
            if abs(v - lp) <= 1e-9 * max(abs(lp), 1e-300):
                msg += f". It equals {text}"
                break
        num = den = 0.0
        for pname, g in ref.grads.items():
            w = _param_whole(recs, layout, plan, pname, "grads")[0][0]
            if g is not None and w is not None:
                num += float((w.double() * g.double()).sum())
                den += float((g.double() ** 2).sum())
        if den > 0:
            msg += (f"; the parameter gradients carry the same error ({num / den:.6g} times the single device "
                    "gradients by a least squares fit)")
        F.append(Finding("TS682", msg, collective="loss normalization", error=le))
        return F, compared, worst
    # 4. backward: output gradients of leaves
    leaves = [n for n in dict.fromkeys(ref.order) if n != "<model>" and not list(mods[n].children())]
    bad_grad = {}
    for name in leaves:
        ncalls = len(ref.calls[name])
        for c in range(ncalls):
            for k, rg in enumerate(ref.calls[name][c].grad):
                if rg is None:
                    continue
                whole, dis = _assemble(_module_slots(recs, layout, name, c, k, ncalls, "grad"), layout, rg,
                                       _hint(recs, name, c))
                if whole is None:
                    continue
                compared["output gradients"] += 1
                e = max(_rel(whole, rg), dis)
                worst["backward"] = max(worst["backward"], e)
                if e > TOL:
                    bad_grad[name] = max(bad_grad.get(name, 0.0), e)
    if bad_grad:
        last = max(bad_grad, key=lambda n: leaves.index(n))
        nxt = next((n for n in leaves[leaves.index(last) + 1:] if n not in bad_grad), None)
        if nxt is None:
            F.append(Finding("TS685", f"the output gradient of {last} differs (error {bad_grad[last]:.1e}) and it "
                                      "is the last module: the gradient of the loss itself is wrong",
                             module=last, collective="loss normalization", error=bad_grad[last]))
        else:
            st = plan.get(nxt)
            why = ""
            coll = None
            if st is not None and st.kind == "column":
                coll = f"{nxt} input gradient {'reduce scatter' if layout.sp else 'all reduce'} (backward)"
                why = (": a column layer's ranks each hold a slice of the weight, so each computes part of the "
                       "input gradient; the backward sum over tp is missing or wrong")
            F.append(Finding("TS685", f"{nxt} ({type(mods[nxt]).__name__}, {sty(nxt)}) passes a wrong input gradient "
                                      f"back: the output gradient of {last} differs (error {bad_grad[last]:.1e}) "
                                      f"while every module after it has the right one{why}",
                             module=nxt, collective=coll, error=bad_grad[last]))
        return F, compared, worst
    # 5. parameter gradients
    gp, gr, bad_params = {}, {}, []
    mod_scale: Dict[str, float] = {}
    for pname, g in ref.grads.items():
        if g is not None:
            mname = recs[0].owner[pname][0]
            mod_scale[mname] = max(mod_scale.get(mname, 0.0), float(g.abs().max()))
    floor = lambda p: 1e-5 * mod_scale.get(recs[0].owner[p][0], 0.0)  # noqa: E731
    for pname in ref.grads:
        if ref.grads[pname] is None:
            continue
        wholes, tdis = _param_whole(recs, layout, plan, pname, "grads")
        if wholes[0] is None:
            continue
        compared["parameter gradients"] += 1
        ddis = max((_rel(w, wholes[0], floor(pname)) for w in wholes[1:]), default=0.0)
        e = _rel(wholes[0], ref.grads[pname], floor(pname))
        worst["gradients"] = max(worst["gradients"], e, tdis, ddis)
        gp[pname], gr[pname] = wholes[0], ref.grads[pname]
        mname, _, short = recs[0].owner[pname]
        if tdis > TOL:
            pieces = [recs[j].grads[pname] for j in range(layout.tp)]
            summed = sum(pieces[1:], pieces[0].clone())
            se = _rel(summed, ref.grads[pname], floor(pname))
            part = recs[0].seq_partial and pname in recs[0].seq_partial
            msg = f"{pname} ({sty(mname)}) gradients differ across tp ranks by {tdis:.1e}"
            if se <= TOL:
                msg += (f"; their sum over tp equals the single device gradient (error {se:.1e}): each rank holds "
                        f"only {'the part from its sequence shard' if part else 'its own part'}, and the gradient all reduce over "
                        "tp for this replicated weight is missing")
            F.append(Finding("TS684", msg, module=mname or None, collective=f"{pname} gradient all reduce over tp",
                             error=tdis))
        elif ddis > TOL:
            F.append(Finding("TS684", f"{pname} gradients differ across dp replicas by {ddis:.1e}: the gradient "
                                      "all reduce over dp does not cover it", module=mname or None,
                             collective="gradient bucket all reduce over dp", error=ddis))
        elif e > TOL:
            bad_params.append((pname, e))
    if F:
        return F, compared, worst
    if bad_params:
        names = [p for p, _ in bad_params]
        num = sum(float((gp[p].double() * gr[p].double()).sum()) for p in names)
        den = sum(float((gr[p].double() ** 2).sum()) for p in names)
        k = num / den if den > 0 else math.nan
        res = math.sqrt(sum(float(((gp[p].double() - k * gr[p].double()) ** 2).sum()) for p in names) / den) if den > 0 else math.inf
        # one factor over every gradient (not only the flagged ones): a scaling, not a module
        tot = sum(float((gr[p].double() ** 2).sum()) for p in gp)
        res_all = math.sqrt(sum(float(((gp[p].double() - k * gr[p].double()) ** 2).sum()) for p in gp) / tot)             if tot > 0 else math.inf
        allp = res_all < 1e-6
        facs = _match_factor(k, layout) if (allp and res < 1e-6) else []
        if len(facs) == 1:
            coll, cause = _FACTOR_CAUSE[facs[0]]
            F.append(Finding("TS683", f"every parameter gradient is {k:.6g} times the single device gradient ({facs[0]}, "
                                      f"fit residual {res:.1e}) while every module output and output gradient "
                                      f"matches: {cause}", collective=coll, error=max(e for _, e in bad_params)))
        elif facs:
            causes = "; or ".join(f"{f}: {_FACTOR_CAUSE[f][1]}" for f in facs)
            F.append(Finding("TS683", f"every parameter gradient is {k:.6g} times the single device gradient (fit "
                                      f"residual {res:.1e}), which is {' or '.join(facs)} in this layout: {causes}. "
                                      "Rerun with values that differ to tell them apart",
                             collective=" or ".join(dict.fromkeys(_FACTOR_CAUSE[f][0] for f in facs)),
                             error=max(e for _, e in bad_params)))
        elif allp and res < 1e-6:
            F.append(Finding("TS683", f"every parameter gradient is {k:.6g} times the single device gradient (fit "
                                      f"residual {res:.1e}), a factor that is none of dp, ga, tp or their inverses",
                             collective="gradient scaling", error=max(e for _, e in bad_params)))
        else:
            by_mod: Dict[str, float] = {}
            for p, e in bad_params:
                by_mod[recs[0].owner[p][0]] = max(by_mod.get(recs[0].owner[p][0], 0.0), e)
            for mname, e in by_mod.items():
                F.append(Finding("TS686", f"{mname} ({sty(mname)}): its parameter gradients differ (error {e:.1e}) "
                                          "while its output and output gradient match: its weight gradient or "
                                          "its gradient sync is wrong", module=mname or None, error=e))
        return F, compared, worst
    # 6. update
    bad_upd = []
    drift = []
    for pname in ref.updated:
        wholes, tdis = _param_whole(recs, layout, plan, pname, "updated")
        if wholes[0] is None:
            continue
        compared["updated weights"] += 1
        ddis = max((_rel(w, wholes[0]) for w in wholes[1:]), default=0.0)
        e = _rel(wholes[0], ref.updated[pname])
        worst["update"] = max(worst["update"], e)
        if e > TOL:
            bad_upd.append((pname, e))
        if max(tdis, ddis) > TOL:
            drift.append((pname, max(tdis, ddis)))
    if bad_upd:
        p, e = max(bad_upd, key=lambda x: x[1])
        msg = (f"every gradient matches but {len(bad_upd)} updated weights differ (worst {p}, error {e:.1e}): the "
               "optimizer step under the layout is not the single device step")
        coll = "optimizer step"
        if ref.clip_norm is not None:
            norms = sorted({round(r.clip_norm, 12) for r in recs})
            if any(abs(n - ref.clip_norm) > 1e-9 * ref.clip_norm for n in norms):
                msg += (f". Gradient clipping used the norm {', '.join(f'{n:.6g}' for n in norms)} on the ranks, "
                        f"the single device norm is {ref.clip_norm:.6g}: each rank clipped by its local shard's norm; "
                        "sum the squared norms over tp (sharded weights once per shard, replicated ones once)")
                coll = "gradient norm all reduce over tp"
        if layout.shard and drift and coll == "optimizer step":
            msg += (f". The dp replicas disagree on the updated weights (worst {max(d for _, d in drift):.1e}) with "
                    f"state sharded (shard={layout.shard}): each replica updates only its slice, so the parameter "
                    "all gather after the update is missing or wrong")
            coll = "parameter all gather over dp (sharded state)"
        F.append(Finding("TS687", msg, module=None if coll != "optimizer step" else (recs[0].owner[p][0] or None),
                         collective=coll, error=e))
    for p, d in drift:
        F.append(Finding("TS688", f"{p} is replicated but differs between ranks after the step by {d:.1e}",
                         module=recs[0].owner[p][0] or None, error=d))
    return F, compared, worst


def parallel_check(model: nn.Module, inputs, layout, *, targets=None, loss=None, plan: Optional[Dict] = None,
                   optimizer: str = "sgd", lr: float = 0.1, clip: float = 0.0, faults: Sequence[str] = (),
                   ranks: str = "threads", model_file: Optional[str] = None, timeout: float = 60.0,
                   model_name: str = "model", device: str = "cuda", backend: str = "auto", order: str = "fixed",
                   sync: Optional[dict] = None, dist: Optional[dict] = None) -> ParallelReport:
    """Run one training step of `model` single device and under `layout`, both in float64, and blame
    the first module or collective that differs. Raises LayoutError (TS680) for a layout or plan that
    cannot run.

    ranks: "threads" or "gloo" (simulated on this CPU), or "devices": one process per real device
    (realranks.py), `device` "cuda" (one GPU per rank) or "cpu", `backend` auto (NCCL for GPUs where
    torch has it, else gloo), `order` of the sums over ranks ("fixed" or "native", devices.py), `sync`
    the gradient buckets ({"bucket_bytes", "overlap"}; default 25 MiB buckets started during backward).
    `dist` ({"master": "host:port", "world": N, "rank": 0}) spreads the ranks over machines; this
    process is rank 0 and every other machine runs `serve_rank`. Raises DevicesUnavailable (TS821) or
    RendezvousError (TS822) when the ranks cannot start."""
    lay = parse_layout(layout) if isinstance(layout, str) else layout
    unknown = [f for f in faults if f not in FAULTS]
    if unknown:
        raise LayoutError(f"unknown fault {unknown[0]!r}; known: {', '.join(FAULTS)}")
    if ranks not in ("threads", "gloo", "devices"):
        raise LayoutError(f"unknown ranks mode {ranks!r}; use threads, gloo or devices")
    if ranks in ("gloo", "devices") and not model_file:
        raise LayoutError(f"ranks={ranks} starts separate processes, which build the model from its file: pass "
                          "model_file")
    if sync is None and ranks == "devices":
        sync = {"bucket_bytes": 25 << 20, "overlap": True}
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    model = copy.deepcopy(model).double().eval()
    inputs = tuple(_to64(x) for x in inputs)
    loss = loss if loss is not None else MSE()
    with torch.no_grad():
        out0 = model(*inputs)
    if not isinstance(out0, torch.Tensor):
        raise LayoutError("the model must return one tensor (the loss input)")
    if targets is None:
        g = torch.Generator().manual_seed(1234)
        targets = torch.randn(out0.shape, generator=g, dtype=torch.float64)
    targets = _to64(targets)
    B = out0.shape[0] if out0.dim() else 0
    if B == 0 or B % (lay.dp * lay.ga):
        raise LayoutError(f"the batch has {B} rows; dp={lay.dp} times ga={lay.ga} must divide it")
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.dim() and x.shape[0] != B:
            raise LayoutError("every input must have the batch as dim 0")
    if lay.sp:
        if out0.dim() < 2 or any(isinstance(x, torch.Tensor) and x.dim() > lay.seq_dim and x.shape[lay.seq_dim] % lay.tp
                                 for x in inputs):
            raise LayoutError(f"sp=1: the sequence (dim {lay.seq_dim}) must divide by tp={lay.tp}")
    if plan is None:
        p = auto_plan(model, inputs, lay.tp) if lay.tp > 1 else {}
        source = "found automatically: MLP up and down projections only" if p else "none"
    else:
        p, source = plan, "given"
    p = validate_plan(model, p, lay)

    devices: List[str] = []
    if ranks == "devices":
        from .devices import local_devices
        from .realranks import rec_to_cpu, run_local, run_multi_node, to_device

        payload = {"state": model.state_dict(), "inputs": inputs, "targets": targets, "loss": loss, "layout": lay,
                   "plan": p, "optimizer": optimizer, "lr": lr, "clip": clip, "faults": list(faults)}
        ref_dev = "cpu"
        if dist is None:
            devs = local_devices(lay.world, device)
            ref_dev = devs[0]
        elif device.startswith("cuda") and torch.cuda.is_available():
            ref_dev = "cuda"
        t0 = time.perf_counter()
        ref = rec_to_cpu(run_reference(copy.deepcopy(model).to(ref_dev), to_device(inputs, ref_dev),
                                       to_device(targets, ref_dev), loss, optimizer, lr, clip))
        t1 = time.perf_counter()
        if dist is None:
            recs = run_local(model_file, payload, lay.world, devs, timeout, backend, order, sync)
        else:
            if int(dist.get("world", 0)) != lay.world or int(dist.get("rank", 0)) != 0:
                raise LayoutError(f"dist: the layout needs {lay.world} ranks and this process must be rank 0; got "
                                  f"world {dist.get('world')} rank {dist.get('rank')}")
            recs = run_multi_node(copy.deepcopy(model), model_file, payload, 0, lay.world, dist["master"], device,
                                  timeout, backend, order, sync)
        devices = [r.device or f"rank {r.rank}: did not start" for r in recs]
        devices.append(f"reference on {ref_dev}; sums over ranks in {order} order; gradient buckets "
                       f"{sync.get('bucket_bytes', 0) >> 20} MiB, overlap with backward "
                       f"{'on' if sync.get('overlap') else 'off'}")
    else:
        t0 = time.perf_counter()
        ref = run_reference(copy.deepcopy(model), inputs, targets, loss, optimizer, lr, clip)
        t1 = time.perf_counter()
    if ranks == "threads":
        recs = _run_threads(model, inputs, targets, loss, lay, p, optimizer, lr, clip, faults, timeout)
    elif ranks == "gloo":
        recs = _run_gloo(model_file, model.state_dict(), inputs, targets, loss, lay, p, optimizer, lr, clip, faults,
                         timeout)
    t2 = time.perf_counter()
    findings, compared, worst = _diagnose(ref, recs, lay, p, model, out0, targets, loss)
    t3 = time.perf_counter()
    if ranks == "devices":
        findings.append(Finding("TS820", f"{lay.world} real ranks, one process each: " + "; ".join(devices)))
        models = sorted({r.device.split("(")[-1].split(",")[0] for r in recs if r.device})
        if len(models) > 1 or (order == "native" and lay.world > 2):
            why = (f"the ranks run on different GPU models ({', '.join(models)}), whose kernels may round differently"
                   if len(models) > 1 else f"the native all reduce over {lay.world} ranks does not add in rank order")
            findings.append(Finding("TS825", f"{why}: this float64 check is unaffected, but a float32 job on these "
                                              "ranks is not bitwise equal to one device; compare it within tolerance"))
    else:
        findings.append(Finding("TS690", f"{lay.world} ranks simulated with {ranks} on this CPU"))
    mods = dict(model.named_modules())
    sources = {f.module: s for f in findings if f.module and f.module in mods for s in [_source(mods[f.module])] if s}
    return ParallelReport(layout=lay, plan={k: v.text() for k, v in p.items()}, plan_source=source, ranks=ranks,
                          findings=findings, loss_reference=ref.loss, loss_parallel=recs[0].loss, compared=compared,
                          worst=worst, timings={"reference": t1 - t0, "parallel": t2 - t1, "diagnosis": t3 - t2},
                          collectives=len(recs[0].log), faults=list(faults), model_name=model_name, sources=sources,
                          devices=devices)


def serve_rank(model_file: str, dist: dict, *, device: str = "cuda", backend: str = "auto", order: str = "fixed",
               sync: Optional[dict] = None, timeout: float = 120.0) -> str:
    """A rank other than 0 of a layout check spread over machines (`truestar parallel --dist-rank R`):
    joins rank 0 at dist["master"], receives the weights and the batch from it, runs its step and
    sends its record back. Returns a one line summary."""
    from ..cli import _load
    from .realranks import run_multi_node

    model, _, _ = _load(model_file)
    rank, world = int(dist["rank"]), int(dist["world"])
    if rank == 0:
        raise LayoutError("serve_rank is for ranks other than 0; rank 0 runs parallel_check with dist=")
    run_multi_node(model, model_file, None, rank, world, dist["master"], device, timeout, backend, order,
                   sync if sync is not None else {"bucket_bytes": 25 << 20, "overlap": True})
    return f"rank {rank} of {world}: step done, record sent to rank 0 at {dist['master']}"
