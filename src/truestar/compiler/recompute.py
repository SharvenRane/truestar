"""Recomputation as a verified compiler pass (docs/MEMORY.md).

A training program (forward, loss and backward in one IR program, ``truestar.train``) keeps forward
values alive until the backward reads them. This pass chooses values to drop after their last
forward reader and compute again right before their next reader, until the planned peak fits the
memory budget.

Why the recomputed value is the same bits: the pass works on the compiled program, not on the IR.
A recomputed value is the same compiled step run again: the same kernel binary, the same launch
config (chosen per input shape signature, which does not change), reading inputs that are either
kept (IR values are never mutated, and in place reuse only writes over buffers no later step reads,
which now includes the recompute) or recomputed the same way. Truestar kernels have no atomics, so
the result is bitwise equal. Recomputing at the IR level instead would let fusion put the copy into
a different kernel, with a different summation order. Library steps (cuDNN convolutions) are never
recomputed: cuDNN may pick another algorithm when free memory differs.

Verified: on every check run (the first call, or every call with verify="all") the stored value is
copied aside before the step that drops it, and the recomputed value is compared with it bit for bit;
a difference raises RecomputeMismatchError (TS421) naming the op and line, and nothing is returned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..ir.core import IRError
from ..kernel import ir as kir
from . import memory as M


def _work(ck, bindings: Dict[str, int]) -> int:
    """Rough cost of a kernel: output elements times the length of its reductions (at least 1)."""
    o = ck.kernel.outputs[0]
    n = 1
    for a in o.axes:
        n *= M.eval_ksize(a.size, bindings)
    red = 0
    for node in kir.walk(o.expr):
        if isinstance(node, kir.Reduce):
            red += M.eval_ksize(node.axis.size, bindings)
    return n * max(1, red)


class _Trial:
    """The plan with another step order, without touching the real plan."""

    def __init__(self, plan, steps):
        self.steps = steps
        self.in_slots, self.template, self.out_slots = plan.in_slots, plan.template, plan.out_slots
        self.opts, self.alias_out, self.path = plan.opts, plan.alias_out, plan.path


def recomputable(st, facts) -> bool:
    """Steps whose rerun gives the same bits: Truestar kernels (deterministic: no atomics, launch
    config fixed per shape), layout copies and dtype adapters (exact copies and casts). Views are not
    worth it (they hold no storage of their own); library, index, host and fallback steps are never
    recomputed (cuDNN may pick another algorithm when free memory differs)."""
    from . import runtime as R
    if type(st) is R.KernelStep:
        return True
    if type(st) is R.AdapterStep:
        return True
    if type(st) is R.ImplStep:      # a copy, or a view (no storage: only as part of a chain)
        return True
    return False


def apply_recompute(cp, budget: int, dims: Dict[str, int], max_rounds: int = 2000,
                exact_per_round: int = 6) -> List[Dict[str, Any]]:
    """Insert recompute steps into ``cp.plan`` until the planned total at ``dims`` fits ``budget``
    (or nothing more helps; the caller's budget check then raises). Returns what was recomputed.

    Greedy over the values live at the planned peak: each candidate is a value and, when it helps,
    the chain of producers it would otherwise keep alive (up to three levels, like a checkpointed
    segment). Candidates are ranked by bytes freed per unit of recompute work; the best few are
    planned exactly and the best exact gain per work is taken."""
    from . import runtime as R
    plan = cp.plan
    if not plan.opts.memory_reuse:
        raise IRError("recompute needs memory_reuse=True: a value that is never freed cannot be recomputed later")
    values, metas = M.input_metas(cp, dims)
    tab = plan.table.evaluate(tuple(values[s] for s in plan.table.symbols))
    base_facts, base_env = M.dry_run(plan, tab, values, metas)
    cov = M.covers_of(base_env)
    facts_of = {id(st): f for st, f in zip(plan.steps, base_facts)}
    dev = plan.opts.device
    resident = sum(M.block_bytes(t.numel() * t.element_size(), dev) for t in M._distinct(plan.template)
                   if t.device.type == dev)
    resident += sum(M.block_bytes(m.numel() * m.element_size(), dev) for m in metas)
    target = budget - resident
    costs: Dict[int, int] = {}
    recompute_facts: Dict[int, Any] = {}     # keyed by the producer's facts (they live as long as the pass)

    def facts_for(steps):
        out = []
        for st in steps:
            if type(st) is R.RecomputeStep:   # new storages of the producer's outputs, never in place
                p = facts_of[id(st.producer)]
                f = recompute_facts.get(id(p))
                if f is None:      # the same writes: a view again, a copy or kernel output new
                    f = recompute_facts[id(p)] = M.StepFacts(list(p.writes))
            else:
                f = facts_of[id(st)]
            out.append(f)
        return out

    def evaluate(steps, live=False):
        t = _Trial(plan, steps)
        return M.walk(t, facts_for(steps), M.schedule(t, True), donated=False, clone_outputs=cp._clone,
                      want_live=live, covers=cov)

    def out_slot(st):
        return st.dst if type(st) is R.AdapterStep else st.out

    def out_bytes(st):
        return sum(b for _, b, _ in facts_of[id(st)].writes)

    def work(st):
        k = id(st)
        if k not in costs:
            costs[k] = _work(st.ck, values) if type(st) is R.KernelStep else max(1, out_bytes(st) // 4)
        return costs[k]

    steps = list(plan.steps)
    w = evaluate(steps, live=True)
    chosen: List[Dict[str, Any]] = []
    outs = set(plan.out_slots)
    rounds = 0
    names = None
    while w.peak > target and rounds < max_rounds:
        rounds += 1
        peak_at = w.peak_step
        live, io = M.ranges(_Trial(plan, steps))
        producer: Dict[int, int] = {}
        for i, st in enumerate(steps):
            if type(st) is not R.RecomputeStep and recomputable(st, facts_of[id(st)]):
                producer[out_slot(st)] = i
        reads_of: Dict[int, List[int]] = {}
        for i, (reads, _) in enumerate(io):
            for s in reads:
                reads_of.setdefault(s, []).append(i)

        def alive_at(q, at):
            return any(a < at <= b for a, b in live.get(q, ()))

        def chain(i, rb, depth, seen):
            """Producer steps to run again before rb, dependencies first."""
            pre = []
            for q in io[i][0]:
                j = producer.get(q)
                if depth > 0 and j is not None and q not in outs and q not in seen and not alive_at(q, rb):
                    seen.add(q)
                    pre += chain(j, rb, depth - 1, seen)
            return pre + [steps[i]]

        at_peak = {s for slots, _ in w.live_at_peak for s in slots}
        cands = []
        for s in at_peak:
            i = producer.get(s)
            if i is None or s in outs or i >= peak_at:
                continue
            st = steps[i]
            if any(x in outs for x in io[i][1]):
                continue
            after = [r for r in reads_of.get(s, []) if r > peak_at]
            if not after:
                continue
            rb = after[0]
            last = None
            for depth in (0, 1, 2, 3, 4, 5):
                prods = chain(i, rb, depth, {s})
                if last is not None and len(prods) == len(last):
                    break
                last = prods
                # estimate: bytes no longer stored over the peak, per unit of work
                est = sum(out_bytes(p) for p in prods if out_slot(p) in at_peak)
                if est <= 0:
                    continue
                cands.append((est / sum(work(p) for p in prods), rb, prods))
        cands.sort(key=lambda c: -c[0])
        best = None
        for k, (_, rb, prods) in enumerate(cands):
            if best is not None and k >= exact_per_round:
                break
            trial = steps[:rb] + [R.RecomputeStep(p) for p in prods] + steps[rb:]
            tw = evaluate(trial)
            gain = w.peak - tw.peak
            if gain <= 0:
                continue
            score = gain / sum(work(p) for p in prods)
            if best is None or score > best[0]:
                best = (score, rb, trial, tw, prods)
        if best is None:
            break
        _, rb, steps, w, prods = best
        w = evaluate(steps, live=True)
        if names is None:
            names = M._names(plan)
        for st in prods:
            name, src = names.get(out_slot(st), (st.name, ""))
            chosen.append({"value": name, "step": st.name, "kind": st.kind, "ops": st.ops(), "source": src,
                           "bytes": out_bytes(st), "recomputed_before": steps[rb + len(prods)].name,
                           "work": work(st)})
    if chosen:
        plan.steps = steps
        plan.set_schedule(M.schedule(plan, True))
    return chosen
