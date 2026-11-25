"""Rematerialization chosen by a minimum cut, and host offload with a fixed overlap schedule.

docs/MEMORY.md, "Min cut recompute and host offload". Research item 4.5.

**Min cut.** The greedy pass (recompute.py) adds recompute chains one at a time while the planned
peak is over budget. This pass decides every forward value at once. Let b be the step at the
planned peak (the end of the forward part of a training program). Every value written by a step
before b becomes two nodes, v_in and v_out, joined by an edge whose capacity is the bytes the value
holds (its own storage, or its base's for a view): cutting it means the value is stored from the
forward part into the backward part. An edge from the source to v_in with capacity lambda times the
recompute work of v's step is cut when v is recomputed. A reader edge u_out -> v_in of infinite
capacity says recomputing v needs u (stored or recomputed); v_out -> sink of infinite capacity says
a step after b reads v. Steps that may not be rerun (library, index, host, random, fallback) get an
infinite source edge: they are stored. Inputs and params are resident and free. So a cut is exactly
a choice "store or recompute" for every value that backward needs, directly or through a
recompute, and its capacity is stored bytes plus lambda times recomputed work. Only what backward
needs is replayed: replay stops at the last value it reads.

Stored bytes are a proxy for the peak, so the pass solves the cut for a sweep of lambda and plans
each distinct cut exactly with the planner's walk; the cut that fits the budget with the least
recompute work is taken (if none fits, the one with the lowest peak, and the budget check then
raises TS420 as before). Each recomputed value is produced again right before the first step that
needs it (its first reader after b, or the recompute of a value that reads it), by the same compiled
step, so the bitwise argument and the TS421 check of recompute.py hold unchanged.

**Offload.** A value stored across b may instead leave the device: an offload step starts a copy to
pinned host memory on a side stream right after its last forward reader, a fence step one step
later makes the main stream wait for that copy and drops the device buffer, a reload step starts
the copy back on the side stream `lookahead` steps before the first backward reader, and a wait
step right before that reader makes the main stream wait for it. All four positions are fixed when
the program is compiled, so the overlap does not depend on timing: the planner counts the device
buffer until the fence and from the reload, and the bits are a copy's bits. Check runs compare
every reloaded value with a copy of the stored one (TS1112). CUDA only (TS1113 elsewhere).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..ir.core import IRError
from . import memory as M
from .recompute import _Trial, _work, recomputable

INF = float("inf")


# ----------------------------------------------------------------------------- max flow (Dinic)


class _Flow:
    def __init__(self, n: int) -> None:
        self.n = n
        self.adj: List[List[int]] = [[] for _ in range(n)]
        self.to: List[int] = []
        self.cap: List[float] = []

    def add(self, u: int, v: int, c: float) -> None:
        self.adj[u].append(len(self.to))
        self.to.append(v)
        self.cap.append(c)
        self.adj[v].append(len(self.to))
        self.to.append(u)
        self.cap.append(0.0)

    def _bfs(self, s: int, t: int) -> Optional[List[int]]:
        level = [-1] * self.n
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if self.cap[e] > 1e-12 and level[self.to[e]] < 0:
                    level[self.to[e]] = level[u] + 1
                    q.append(self.to[e])
        return level if level[t] >= 0 else None

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        while True:
            level = self._bfs(s, t)
            if level is None:
                return flow
            it = [0] * self.n

            def dfs(u: int, f: float) -> float:
                if u == t:
                    return f
                while it[u] < len(self.adj[u]):
                    e = self.adj[u][it[u]]
                    v = self.to[e]
                    if self.cap[e] > 1e-12 and level[v] == level[u] + 1:
                        d = dfs(v, min(f, self.cap[e]))
                        if d > 0:
                            self.cap[e] -= d
                            self.cap[e ^ 1] += d
                            return d
                    it[u] += 1
                return 0.0

            while True:
                f = dfs(s, INF)
                if f <= 0:
                    break
                if f == INF:
                    raise IRError("internal: the remat graph has a path of infinite capacity (a backward value "
                                  "that can be neither stored nor recomputed)")
                flow += f

    def source_side(self, s: int) -> Set[int]:
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if self.cap[e] > 1e-12 and self.to[e] not in seen:
                    seen.add(self.to[e])
                    q.append(self.to[e])
        return seen


def min_cut(nodes: Sequence[int], store: Dict[int, float], work: Dict[int, float], fixed: Set[int],
            reads: Dict[int, List[int]], needed: Set[int], lam: float) -> Set[int]:
    """The values to recompute: a minimum cut of the store or recompute graph described above.
    `store[v]` bytes, `work[v]` recompute work, `fixed` values that must be stored, `reads[v]` the
    graph values v's step reads, `needed` the values a step after the boundary reads."""
    idx = {v: k for k, v in enumerate(nodes)}
    n = len(nodes)
    src, snk = 2 * n, 2 * n + 1
    g = _Flow(2 * n + 2)
    for v in nodes:
        k = idx[v]
        g.add(2 * k, 2 * k + 1, store[v])
        g.add(src, 2 * k, INF if v in fixed else lam * work[v])
        if v in needed:
            g.add(2 * k + 1, snk, INF)
        for u in reads.get(v, ()):
            if u in idx:
                g.add(2 * idx[u] + 1, 2 * k, INF)
    g.max_flow(src, snk)
    side = g.source_side(src)
    return {v for v in nodes if 2 * idx[v] not in side}


# ----------------------------------------------------------------------------- the pass


def _setup(cp, dims):
    from . import runtime as R
    plan = cp.plan
    if not plan.opts.memory_reuse:
        raise IRError("recompute needs memory_reuse=True: a value that is never freed cannot be recomputed later")
    values, metas = M.input_metas(cp, dims)
    tab = plan.table.evaluate(tuple(values[s] for s in plan.table.symbols))
    facts, env = M.dry_run(plan, tab, values, metas)
    cov = M.covers_of(env)
    dev = plan.opts.device
    resident = sum(M.block_bytes(t.numel() * t.element_size(), dev) for t in M._distinct(plan.template)
                   if t.device.type == dev)
    resident += sum(M.block_bytes(m.numel() * m.element_size(), dev) for m in metas)
    return R, plan, values, facts, cov, resident


class _Evaluator:
    """Plans a candidate step list exactly (the same walk the budget check uses)."""

    def __init__(self, cp, plan, facts, cov) -> None:
        from . import runtime as R
        self.R, self.cp, self.plan, self.cov = R, cp, plan, cov
        self.facts_of = {id(st): f for st, f in zip(plan.steps, facts)}
        self.extra: Dict[int, M.StepFacts] = {}

    def facts_for(self, steps):
        R = self.R
        out = []
        for st in steps:
            if type(st) is R.RecomputeStep:
                p = self.facts_of[id(st.producer)]
                f = self.extra.get(id(p))
                if f is None:
                    f = self.extra[id(p)] = M.StepFacts(list(p.writes))
            elif id(st) in self.facts_of:
                f = self.facts_of[id(st)]
            else:
                f = self.extra[id(st)] = st.facts(self.facts_of, self.extra) if hasattr(st, "facts") else M.StepFacts([])
                self.facts_of[id(st)] = f
            out.append(f)
        return out

    def __call__(self, steps, live=False):
        t = _Trial(self.plan, steps)
        return M.walk(t, self.facts_for(steps), M.schedule(t, True), donated=False, clone_outputs=self.cp._clone,
                      want_live=live, covers=self.cov)


def _out_slots(R, st) -> List[int]:
    return [st.dst] if type(st) is R.AdapterStep else M.step_io(st, with_check=False)[1]


def mincut_recompute(cp, budget: int, dims: Dict[str, int], lambdas: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """Insert recompute steps chosen by the minimum cut until the planned peak at `dims` fits `budget`.
    Returns what is recomputed (the same entries as recompute.apply_recompute), plus one summary
    entry under cp.remat_summary."""
    R, plan, values, facts, cov, resident = _setup(cp, dims)
    target = budget - resident
    ev = _Evaluator(cp, plan, facts, cov)
    steps = list(plan.steps)
    w0 = ev(steps)
    cp.remat_summary = {"method": "mincut", "baseline_peak": w0.peak, "target": target, "cuts": 0}
    if w0.peak <= target:
        return []
    b = w0.peak_step
    live, io = M.ranges(_Trial(plan, steps))
    outs = set(plan.out_slots)
    resident_slots = set(plan.in_slots) | {i for i, t in enumerate(plan.template) if t is not None}
    producer: Dict[int, int] = {}
    for i, st in enumerate(steps[:b]):
        for s in io[i][1]:
            producer[s] = i
    view_of: Dict[int, int] = {}
    nbytes: Dict[int, int] = {}
    for i in range(b):
        for slot, nb, src in facts[i].writes:
            nbytes[slot] = nb
            if src is not None:
                view_of[slot] = src

    def base(s):
        seen = 0
        while s in view_of and seen < 64:
            s = view_of[s]
            seen += 1
        return s

    nodes = [s for s in producer if s not in outs and s not in resident_slots]
    nodeset = set(nodes)
    store = {s: float(nbytes.get(base(s), nbytes.get(s, 0))) for s in nodes}
    costs: Dict[int, float] = {}
    fixed: Set[int] = set()
    reads: Dict[int, List[int]] = {}
    for s in nodes:
        st = steps[producer[s]]
        if not recomputable(st, facts[producer[s]]) or any(x in outs for x in io[producer[s]][1]):
            fixed.add(s)
        k = id(st)
        if k not in costs:
            costs[k] = float(_work(st.ck, values)) if type(st) is R.KernelStep else \
                float(max(1, sum(nb for _, nb, _ in facts[producer[s]].writes) // 4))
        reads[s] = [u for u in io[producer[s]][0] if u in nodeset]
    work = {s: costs[id(steps[producer[s]])] for s in nodes}
    readers_after: Dict[int, int] = {}
    for i in range(b + 1, len(steps)):
        for s in io[i][0]:
            if s in nodeset and s not in readers_after:
                readers_after[s] = i
    needed = set(readers_after)
    if lambdas is None:
        ratios = sorted(store[s] / work[s] for s in nodes if s not in fixed and work[s] > 0 and store[s] > 0)
        mid = ratios[len(ratios) // 2] if ratios else 1.0
        lambdas = [0.0] + [mid * 2.0 ** k for k in range(-24, 25)]
    tried: Dict[frozenset, Tuple[Any, ...]] = {}
    best = None       # (fits, work, peak, steps, chosen set)
    lowest = None
    for lam in lambdas:
        rec = frozenset(min_cut(nodes, store, work, fixed, reads, needed, lam if lam > 0 else 1e-30))
        if rec in tried:
            continue
        trial = _insert(R, steps, rec, producer, readers_after, reads, nodeset, io, b)
        tw = ev(trial)
        total_work = sum(costs[id(steps[i])] for i in {producer[s] for s in rec})
        tried[rec] = (tw.peak, total_work)
        cand = (tw.peak <= target, -total_work, -tw.peak)
        row = (cand, trial, rec, tw.peak, total_work)
        if tw.peak <= target and (best is None or total_work < best[4] or (total_work == best[4] and tw.peak < best[3])):
            best = row
        if lowest is None or tw.peak < lowest[3]:
            lowest = row
    cp.remat_summary.update(cuts=len(tried), boundary_step=b)
    pick = best or lowest
    if pick is None or pick[3] >= w0.peak:
        return []
    _, new_steps, rec, peak, total_work = pick
    if best is not None:
        # the sweep jumps between cuts; store again, most work first, what the budget does not need
        groups = sorted({producer[s] for s in rec}, key=lambda i: -costs[id(steps[i])])
        rec = set(rec)
        refined = 0
        for i in groups:
            trial_rec = _prune({s for s in rec if producer[s] != i}, reads, needed)
            if trial_rec == rec:
                continue
            trial = _insert(R, steps, trial_rec, producer, readers_after, reads, nodeset, io, b)
            tw = ev(trial)
            if tw.peak <= target:
                rec, new_steps, peak = trial_rec, trial, tw.peak
                refined += 1
        total_work = sum(costs[id(steps[i])] for i in {producer[s] for s in rec})
        cp.remat_summary["stored_again_after_cut"] = refined
    cp.remat_summary.update(planned_peak=peak, work=total_work, fits=best is not None)
    plan.steps = new_steps
    plan.set_schedule(M.schedule(plan, True))
    names = M._names(plan)
    chosen = []
    for i in sorted({producer[s] for s in rec}):
        st = steps[i]
        out = _out_slots(R, st)[0]
        name, src = names.get(out, (st.name, ""))
        chosen.append({"value": name, "step": st.name, "kind": st.kind, "ops": st.ops(), "source": src,
                       "bytes": sum(nb for _, nb, _ in facts[i].writes), "work": int(costs[id(st)]),
                       "method": "mincut", "code": "TS1110"})
    return chosen


def _insert(R, steps, rec, producer, readers_after, reads, nodeset, io, b):
    """The step list with every recomputed value's step run again right before its first need."""
    need: Dict[int, int] = {}
    consumers: Dict[int, List[int]] = {}
    for v in rec:
        for u in reads.get(v, ()):
            consumers.setdefault(u, []).append(v)
    # reverse topological: a value's need is before every recomputed consumer's need
    for v in sorted(rec, key=lambda s: -producer[s]):
        t = readers_after.get(v, math.inf)
        for c in consumers.get(v, ()):
            if c in rec:
                t = min(t, need[c])
        need[v] = t
    at: Dict[int, List[int]] = {}
    done: Set[int] = set()
    for v in sorted(rec, key=lambda s: producer[s]):
        i = producer[v]
        if i in done or need[v] == math.inf:
            continue
        done.add(i)
        # a step writing several recomputed slots runs once, before the earliest need among them
        t = min(need[s] for s in rec if producer[s] == i)
        at.setdefault(int(t), []).append(i)
    out = []
    for k, st in enumerate(steps):
        for i in sorted(at.get(k, ())):
            out.append(R.RecomputeStep(steps[i]))
        out.append(st)
    return out


# ----------------------------------------------------------------------------- host offload

LOOKAHEAD = 4        # steps between starting a reload and the reader that waits for it


def offload_values(cp, budget: int, dims: Dict[str, int], lookahead: int = LOOKAHEAD) -> List[Dict[str, Any]]:
    """Offload values stored across the peak, largest first, until the planned peak fits `budget`.
    Positions are fixed here: offload after the last reader before the boundary, fence one step
    later, reload `lookahead` steps before the first reader after it, wait right before that reader."""
    R, plan, values, facts, cov, resident = _setup(cp, dims)
    if plan.opts.device != "cuda":
        raise IRError(f"[TS1113] host offload needs a CUDA device (pinned host memory and a copy stream); "
                      f"this program is compiled for {plan.opts.device}")
    target = budget - resident
    ev = _Evaluator(cp, plan, facts, cov)
    steps = list(plan.steps)
    w = ev(steps)
    cp.offload_summary = {"baseline_peak": w.peak, "target": target, "offloaded": 0}
    if w.peak <= target:
        return []
    b = w.peak_step
    live, io = M.ranges(_Trial(plan, steps))
    outs = set(plan.out_slots)
    resident_slots = set(plan.in_slots) | {i for i, t in enumerate(plan.template) if t is not None}
    nbytes: Dict[int, int] = {}
    view_children: Dict[int, List[int]] = {}
    for f in facts:
        for slot, nb, src in f.writes:
            if src is None:
                nbytes[slot] = nb
            else:
                view_children.setdefault(src, []).append(slot)
    readers: Dict[int, List[int]] = {}
    for i, (rd, _) in enumerate(io):
        for s in rd:
            readers.setdefault(s, []).append(i)

    def across(s):
        return any(a < b < e for a, e in live.get(s, ()))

    def has_live_view(s, depth=0):
        return depth < 16 and any(across(c) or has_live_view(c, depth + 1) for c in view_children.get(s, ()))

    cands = []
    for s, rs in live.items():
        if s in outs or s in resident_slots or nbytes.get(s, 0) <= 0 or len(rs) != 1 or not across(s):
            continue
        if has_live_view(s):
            continue
        before = [r for r in readers.get(s, []) if r <= b]
        after = [r for r in readers.get(s, []) if r > b]
        if not after:
            continue
        last_fwd = max(before) if before else rs[0][0]
        if after[0] - last_fwd < 3:      # no room for offload, fence, reload and wait between the two
            continue
        cands.append((nbytes[s], s, last_fwd, after[0]))
    cands.sort(key=lambda c: -c[0])
    chosen: List[Tuple[int, int, int, int]] = []
    names = M._names(plan)
    best_steps, best_w = steps, w
    for nb, s, last_fwd, first_bwd in cands:
        trial_set = chosen + [(nb, s, last_fwd, first_bwd)]
        trial = _with_offloads(R, plan, steps, trial_set, lookahead)
        tw = ev(trial)
        if tw.peak < best_w.peak:
            chosen, best_steps, best_w = trial_set, trial, tw
            if tw.peak <= target:
                break
    if not chosen:
        return []
    plan.steps = best_steps
    plan.set_schedule(M.schedule(plan, True))
    cp.offload_summary.update(offloaded=len(chosen), planned_peak=best_w.peak, fits=best_w.peak <= target,
                              bytes=sum(c[0] for c in chosen))
    out = []
    for nb, s, last_fwd, first_bwd in chosen:
        name, src = names.get(s, (f"slot {s}", ""))
        out.append({"value": name, "step": steps[last_fwd].name, "kind": "offload", "ops": [], "source": src,
                    "bytes": nb, "method": "offload", "code": "TS1111",
                    "reloaded_before": steps[first_bwd].name})
    return out


def _with_offloads(R, plan, steps, chosen, lookahead):
    """The step list with offload, fence, reload and wait steps for every chosen value. Host slots
    are new entries at the end of the plan's slot table."""
    after: Dict[int, List[Any]] = {}
    before: Dict[int, List[Any]] = {}
    host_of = getattr(plan, "_host_slots", None)
    if host_of is None:
        host_of = plan._host_slots = {}
    for nb, s, last_fwd, first_bwd in chosen:
        h = host_of.get(s)
        if h is None:
            h = host_of[s] = len(plan.template)
            plan.template.append(None)
        off = R.OffloadStep(s, h, f"%{s}")
        fence = R.FenceStep(off)
        rel = R.ReloadStep(off, plan.opts.torch_device)
        wait = R.WaitStep(rel)
        after.setdefault(last_fwd, []).append(off)
        after.setdefault(min(last_fwd + 1, first_bwd - 1), []).append(fence)
        start = max(last_fwd + 2, first_bwd - lookahead)
        before.setdefault(start, []).append(rel)
        before.setdefault(first_bwd, []).append(wait)
    out = []
    for i, st in enumerate(steps):
        out += before.get(i, [])
        out.append(st)
        out += after.get(i, [])
    return out


def apply_remat(cp, budget: int, dims: Dict[str, int], mode: str) -> List[Dict[str, Any]]:
    """`mode`: "mincut", "offload" or "mincut+offload" (min cut first, then offload what still does
    not fit)."""
    chosen: List[Dict[str, Any]] = []
    if "mincut" in mode:
        chosen += mincut_recompute(cp, budget, dims)
    if "offload" in mode:
        chosen += offload_values(cp, budget, dims)
    return chosen


def _prune(rec: Set[int], reads: Dict[int, List[int]], needed: Set[int]) -> Set[int]:
    """Keep only recomputed values that a step after the boundary reads, or that another kept
    recomputed value reads (directly or through a chain)."""
    keep = {v for v in rec if v in needed}
    stack = list(keep)
    while stack:
        v = stack.pop()
        for u in reads.get(v, ()):
            if u in rec and u not in keep:
                keep.add(u)
                stack.append(u)
    return keep
