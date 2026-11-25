"""``truestar profile``: GPU time of every compiled step, mapped to its IR ops and the user's lines.

Truestar runs every step of a compiled program itself, so it can time each one directly: a CUDA
event before and after every step (kernel, library kernel, fallback, layout copy), recorded from
the runtime's own step loop. Nothing is sampled and no trace is parsed, so:

* every launch is accounted for: the number of timed steps is checked against the number of steps
  the runtime ran, and every event pair is checked to have completed. A shortfall is reported as
  TS510, never dropped (PyTorch's profiler can lose kernel events silently,
  https://github.com/pytorch/pytorch/issues/192021);
* each time belongs to a known step, whose fused IR ops and source lines are known from the
  compile, so time maps to lines exactly, fusion groups included;
* first call costs (compile, Triton compile, verification) are reported apart from steady state.

``crosscheck=True`` also runs the calls under ``torch.profiler`` and compares its kernel count and
total with the events (TS511 when the profiler saw fewer Truestar kernels than were launched).
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from .checks import source_line

FLOP_OPS = ("matmul", "linear", "attention", "conv2d")


class StepTimer:
    """Collects one (start, end) CUDA event pair per step run, from Plan._run_instrumented."""

    def __init__(self, rt=None):
        self.rt = rt
        self.records: List[tuple] = []     # (plan, step, start event, end event, host seconds)
        self.ran = 0                       # steps the runtime ran while timing (the accounting base)
        self._open: Dict[int, tuple] = {}
        self.last = None                   # the event that ends the previous step of this call
        self.target = None                 # id of the one step to time per call, or None for all
        self.runs: Dict[int, int] = {}     # step runs seen, timed or not
        self.order: List[int] = []
        self._pool: List[torch.cuda.Event] = []
        self._used = 0
        self.bytes: Dict[int, int] = {}    # per step: bytes read and written (measured on its first run)

    def _event(self) -> torch.cuda.Event:
        if self._used < len(self._pool):
            e = self._pool[self._used]
        else:
            e = torch.cuda.Event(enable_timing=True)
            self._pool.append(e)
        self._used += 1
        return e

    def reset(self) -> None:
        self.records = []
        self.ran = 0
        self._used = 0

    def start(self, plan, step) -> None:
        if step.kind in ("cond", "loop"):
            return                         # their regions' steps are timed one by one
        if self.target is not None:        # one step per call: its own pair of events
            if id(step) == self.target:
                e = self._event()
                e.record()
                self._open[id(step)] = (e, time.perf_counter())
            return
        if self.last is None:              # the first step of a call: one event before it
            self.last = self._event()
            self.last.record()
        self._open[id(step)] = (self.last, time.perf_counter())

    def stop(self, plan, step, env) -> None:
        """One event after each step; it also starts the next step (steps run back to back on
        the stream), so a call of n steps records n + 1 events, not 2 n."""
        if step.kind in ("cond", "loop"):
            return
        self.runs[id(step)] = self.runs.get(id(step), 0) + 1
        if self.target is not None and id(step) != self.target:
            return
        self.ran += 1
        e0, t0 = self._open.pop(id(step))
        e1 = self._event()
        e1.record()
        self.last = e1
        if id(step) not in self.bytes:
            self.bytes[id(step)] = _bytes(step, env)
        self.records.append((plan, step, e0, e1, time.perf_counter() - t0))


def _shape(plan, t, rt) -> tuple:
    from ..ir.reference import eval_dim
    return tuple(d if isinstance(d, int) else int(eval_dim(d, rt.bindings)) for d in t.type.shape)


def _flops(plan, step, rt) -> Optional[int]:
    total, any_ = 0, False
    for n in step.nodes:
        if n.op not in FLOP_OPS:
            continue
        try:
            out = math.prod(_shape(plan, n.output, rt))
            if n.op in ("matmul", "linear"):
                k = _shape(plan, n.inputs[0], rt)[-1]
                total += 2 * out * k
            elif n.op == "attention":
                q, kk = _shape(plan, n.inputs[0], rt), _shape(plan, n.inputs[1], rt)
                total += 4 * math.prod(q[:-1]) * kk[-2] * q[-1]
            else:
                w = _shape(plan, n.inputs[1], rt)
                total += 2 * out * math.prod(w[1:])
            any_ = True
        except Exception:  # noqa: BLE001 (a FLOP count is informative only)
            return None
    return total if any_ else None


def _bytes(step, env) -> int:
    """Bytes the step read and wrote: its input and output tensors, counted once each."""
    slots = set()
    for name in ("in_slots", "outs", "args"):
        for s in getattr(step, name, None) or []:
            if isinstance(s, int):
                slots.add(s)
    for name in ("out", "src", "dst"):
        s = getattr(step, name, None)
        if isinstance(s, int):
            slots.add(s)
    total = 0
    for s in slots:
        t = env[s] if s < len(env) else None
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def _lines(step) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for n in step.nodes:
        f, line = source_line(n.source)
        text = n.source.split(" in ", 1)[-1] if n.source else ""
        key = (f, line)
        if f is None or key in seen:
            continue
        seen.add(key)
        out.append({"file": f, "line": line, "op": n.op, "text": text.split(": ", 1)[-1] if ": " in text else text})
    return out


def profile_program(cp, inputs: Sequence[torch.Tensor], iters: int = 30, warmup: int = 3,
                    crosscheck: bool = False) -> Dict[str, Any]:
    """Profile a CompiledProgram on CUDA. Returns the profile as a dict (see docs/PROFILE.md)."""
    if cp.opts.device != "cuda":
        raise ValueError("truestar profile times GPU steps; compile the program for 'cuda'")
    # the profile times each step of an ordinary call; an automatic graph replay has no steps to time
    auto, cp.auto_graphs = getattr(cp, "auto_graphs", False), False
    try:
        return _profile_program(cp, inputs, iters, warmup, crosscheck)
    finally:
        cp.auto_graphs = auto


def _profile_program(cp, inputs, iters, warmup, crosscheck):
    t0 = time.perf_counter()
    cp(list(inputs))                                   # first call: Triton compile and the checks
    torch.cuda.synchronize()
    first_call = time.perf_counter() - t0
    first = {"compile_s": cp.compile_seconds, "first_call_s": first_call,
             "verify_s": cp.verify_seconds + cp.kernel_verify_seconds}
    saved = cp.verify
    cp.verify = "off"
    try:
        for _ in range(max(0, warmup)):
            cp(list(inputs))
        torch.cuda.synchronize()
        # 1. uninstrumented, as the program runs: whole call spans (GPU idle gaps while the host
        #    launches included), and the host time to issue one call
        plain, host = [], []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            h = time.perf_counter()
            cp(list(inputs))
            host.append(time.perf_counter() - h)
            b.record()
            plain.append((a, b))
        torch.cuda.synchronize()
        plain_ms = [a.elapsed_time(b) for a, b in plain]
        host_ms = statistics.median(host) * 1000.0
        # The GPU is kept busy while the host issues a call (a sleep kernel queued first), so the
        # spans below are GPU busy time, not launch gaps: what each step itself costs.
        probe = StepTimer()
        _set_timer(cp, inputs, probe)
        h = time.perf_counter()
        cp(list(inputs))                       # host time of an instrumented call
        inst_host_ms = (time.perf_counter() - h) * 1000.0
        _set_timer(cp, inputs, None)
        torch.cuda.synchronize()
        cycles = _sleep_cycles(3.0 * max(host_ms, inst_host_ms) + 0.5)
        busy = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(cycles)
            a.record()
            cp(list(inputs))
            b.record()
            busy.append((a, b))
        torch.cuda.synchronize()
        busy_ms = [a.elapsed_time(b) for a, b in busy]
        # 2. instrumented. Each call times one step (every run of it in that call) with its own
        #    pair of events, and the calls cycle through the steps, so a call carries two extra
        #    events whatever its length (recording an event between every step stretched the
        #    transformer layer's GPU time by 12 percent on this machine).
        probe = StepTimer()
        _set_timer(cp, inputs, probe)
        cp(list(inputs))
        torch.cuda.synchronize()
        targets = list(dict.fromkeys(id(r[1]) for r in probe.records))
        per_call_runs = {k: probe.runs[k] for k in targets}
        timer = StepTimer()
        spans, per_iter = [], []
        orig_bind = cp._bound
        for _ in range(iters):
            for tid in targets:
                timer_iter_start = len(timer.records)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                _set_timer(cp, inputs, timer)
                timer.target = tid
                torch.cuda._sleep(cycles)
                a.record()
                cp(list(inputs))
                b.record()
                spans.append((a, b))
                per_iter.append((timer_iter_start, len(timer.records)))
        timer.expected = sum(per_call_runs.values()) * iters
        _set_timer(cp, inputs, None)
        torch.cuda.synchronize()
        assert cp._bound is orig_bind
        prof = None
        if crosscheck:
            prof = _crosscheck(cp, inputs, iters)
    finally:
        cp.verify = saved
    return _summarize(cp, timer, spans, per_iter, plain_ms, busy_ms, host_ms, first, iters, prof)


_CYCLES_PER_MS: List[float] = []


def _sleep_cycles(ms: float) -> int:
    """GPU clock cycles of torch.cuda._sleep that last about ``ms`` (calibrated once)."""
    if not _CYCLES_PER_MS:
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(1000)
        a.record()
        torch.cuda._sleep(2_000_000)
        b.record()
        torch.cuda.synchronize()
        _CYCLES_PER_MS.append(2_000_000 / max(a.elapsed_time(b), 1e-3))
    return int(min(ms, 200.0) * _CYCLES_PER_MS[0])


def _set_timer(cp, inputs, timer) -> None:
    key = tuple([t.shape for t in inputs])
    rt = cp._bound.get(key)
    if rt is None:
        rt = cp._bind(inputs, key)
    rt.timer = timer
    if timer is not None:
        timer.rt = rt
        timer.last = None


def _crosscheck(cp, inputs, iters: int) -> Dict[str, Any]:
    """Run the calls under torch.profiler and count what it saw (Truestar kernels are the Triton
    functions named kern and kern_loop; the runtime knows how many it launched)."""
    from torch.profiler import ProfilerActivity, profile
    launched = sum(1 for _, s in cp.plan.all_steps() if s.kind == "kernel")
    counter = StepTimer()
    _set_timer(cp, inputs, counter)
    cp(list(inputs))                                  # how many kernel steps one call runs (loops)
    torch.cuda.synchronize()
    _set_timer(cp, inputs, None)
    per_call = sum(1 for r in counter.records if r[1].kind == "kernel")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            cp(list(inputs))
        torch.cuda.synchronize()
    kern, other, kern_us, all_us = 0, 0, 0.0, 0.0
    for ev in p.events():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        dur = ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total
        all_us += dur
        if ev.name in ("kern", "kern_loop"):
            kern += 1
            kern_us += dur
        else:
            other += 1
    expected = per_call * iters
    return {"tool": "torch.profiler", "iterations": iters, "truestar_kernels_launched": expected,
            "truestar_kernels_seen": kern, "dropped": max(0, expected - kern), "other_gpu_events": other,
            "truestar_kernel_ms_per_call": kern_us / 1000.0 / iters, "gpu_ms_per_call": all_us / 1000.0 / iters,
            "kernel_steps_in_program": launched}


def _summarize(cp, timer, spans, per_iter, plain_ms, busy_ms, host_ms, first, iters, prof) -> Dict[str, Any]:
    span_ms = [a.elapsed_time(b) for a, b in spans]
    rows: Dict[int, Dict[str, Any]] = {}
    unaccounted = []
    host_s = 0.0
    call_of = {}
    for c, (lo, hi) in enumerate(per_iter):
        for k in range(lo, hi):
            call_of[k] = c
    for k, (plan, step, e0, e1, host) in enumerate(timer.records):
        ok = False
        try:
            ok = e0.query() and e1.query()
            ms = e0.elapsed_time(e1) if ok else float("nan")
        except (RuntimeError, ValueError):        # an event that was never recorded or did not complete
            ms = float("nan")
        if not ok or not math.isfinite(ms) or ms < 0:
            unaccounted.append({"step": step.name, "region": plan.path})
            continue
        host_s += host
        r = rows.get(id(step))
        if r is None:
            rt = timer.rt
            r = rows[id(step)] = {
                "step": step.name, "kind": step.kind, "region": plan.path,
                "shape": getattr(getattr(step, "ck", None), "plan_kind", None),
                "ops": step.ops(), "values": [o.name for n in step.nodes for o in n.outputs],
                "fused": step.kind == "kernel" and len(step.nodes) > 1,
                "impl": getattr(step, "impl", None) or (getattr(step, "reason", None) and "fallback"),
                "lines": _lines(step), "calls": 0, "ms": [], "bytes": timer.bytes.get(id(step), 0),
                "flops": _flops(plan, step, rt) if rt is not None else None, "first": k}
        r["calls"] += 1
        r["ms"].append(ms)
        c = call_of.get(k, -1)
        r.setdefault("by_call", {})[c] = r.setdefault("by_call", {}).get(c, 0.0) + ms
    # per step: its time in the fastest of the calls that timed it (the GPU is shared; the median
    # is reported next to it)
    for r in rows.values():
        calls = list(r["by_call"].values())
        r["best"] = min(calls)
        r["median"] = statistics.median(calls)
    total_ms = sum(r["best"] for r in rows.values())
    steps = []
    for r in sorted(rows.values(), key=lambda r: -r["best"]):
        per_call = r["best"]
        runs = r["calls"] / max(1, len(r["by_call"]))
        mean = per_call / runs
        r2 = {k: v for k, v in r.items() if k not in ("ms", "first", "by_call", "best", "median")}
        r2["calls"] = int(runs) if runs == int(runs) else runs
        r2["gpu_ms"] = per_call
        r2["gpu_ms_median"] = r["median"]
        r2["gpu_ms_per_launch"] = mean
        r2["share"] = per_call / total_ms if total_ms > 0 else None
        r2["gb_per_s"] = (r["bytes"] / (mean * 1e-3) / 1e9) if mean > 0 else None
        r2["tflop_per_s"] = (r["flops"] / (mean * 1e-3) / 1e12) if r["flops"] and mean > 0 else None
        steps.append(r2)
    lines: Dict[tuple, Dict[str, Any]] = {}
    for r in steps:
        for ln in r["lines"]:
            key = (ln["file"], ln["line"])
            d = lines.setdefault(key, {"file": ln["file"], "line": ln["line"], "text": ln["text"],
                                       "gpu_ms": 0.0, "gpu_ms_shared": 0.0, "steps": []})
            if len(r["lines"]) == 1:
                d["gpu_ms"] += r["gpu_ms"]
            else:
                d["gpu_ms_shared"] += r["gpu_ms"]
            d["steps"].append(r["step"])
    line_list = sorted(lines.values(), key=lambda d: -(d["gpu_ms"] + d["gpu_ms_shared"]))
    no_line = sum(r["gpu_ms"] for r in steps if not r["lines"])
    launched = timer.ran
    expected = getattr(timer, "expected", launched)
    if launched != expected:
        unaccounted.append({"step": f"{expected - launched} step runs expected but not seen", "region": ""})
    timed = launched - len(unaccounted)
    med = statistics.median
    out = {
        "program": cp.program.name, "device": cp.opts.device, "dtype": cp.opts.dtype,
        "gpu": torch.cuda.get_device_name(), "iterations": iters,
        "totals": {"step_gpu_ms": total_ms, "gpu_busy_ms": min(busy_ms), "gpu_busy_ms_median": med(busy_ms),
                   "instrumented_span_ms": med(span_ms), "event_overhead_ms": med(span_ms) - med(busy_ms),
                   "call_span_ms": med(plain_ms), "host_issue_ms": host_ms,
                   "host_bound": host_ms > med(busy_ms), "gpu_ms_without_line": no_line,
                   "steps": len(steps)},
        "accounting": {"steps_run": launched, "steps_expected": expected, "steps_timed": timed,
                       "unaccounted": unaccounted, "method": "one step timed per call, cycling"},
        "first_call": first, "steps": steps, "lines": line_list, "crosscheck": prof,
        "sentinels": sorted(cp.opts.checks)}
    return out


# ----------------------------------------------------------------------------- reports


def to_text(p: Dict[str, Any], top: int = 15) -> str:
    t, acc = p["totals"], p["accounting"]
    lines = [f"truestar profile: {p['program']} on {p['gpu']}, {p['dtype']}, {p['iterations']} calls",
             f"  per call: GPU busy {t['gpu_busy_ms']:.3f} ms, steps sum to {t['step_gpu_ms']:.3f} ms in {t['steps']} "
             f"distinct steps; as run {t['call_span_ms']:.3f} ms, host issue {t['host_issue_ms']:.3f} ms"
             + (" (host bound)" if t["host_bound"] else ""),
             f"  events: {acc['steps_timed']} of {acc['steps_run']} step runs timed"
             + ("" if not acc["unaccounted"] else f", {len(acc['unaccounted'])} UNACCOUNTED (TS510)"),
             f"  first call: compile {p['first_call']['compile_s']:.2f} s, first run {p['first_call']['first_call_s']:.2f} s"
             f" (checks against float64 {p['first_call']['verify_s']:.2f} s), not in the times below",
             "", f"  {'share':>6} {'ms/call':>8} {'calls':>5}  {'kind':<8} {'shape':<11} {'GB/s':>6} {'TFLOP/s':>7}  ops  [line]"]
    for r in p["steps"][:top]:
        where = ", ".join(f"{_short(ln['file'])}:{ln['line']}" for ln in r["lines"][:3]) or "-"
        ops = " + ".join(r["ops"])
        if len(ops) > 48:
            ops = ops[:45] + "..."
        gb = f"{r['gb_per_s']:.0f}" if r["gb_per_s"] is not None else "-"
        tf = f"{r['tflop_per_s']:.1f}" if r["tflop_per_s"] is not None else "-"
        share = f"{100 * r['share']:.1f}%" if r["share"] is not None else "-"
        lines.append(f"  {share:>6} {r['gpu_ms']:8.4f} {r['calls']:>5}  {r['kind']:<8} {str(r['shape'] or r['impl'] or '-')[:11]:<11} "
                     f"{gb:>6} {tf:>7}  {ops}  [{where}]")
    if len(p["steps"]) > top:
        rest = sum(r["gpu_ms"] for r in p["steps"][top:])
        lines.append(f"  ... {len(p['steps']) - top} more steps, {rest:.4f} ms")
    lines += ["", "  by source line (ms per call; shared = in a kernel fused with other lines):"]
    for d in p["lines"][:top]:
        lines.append(f"    {_short(d['file'])}:{d['line']:<5} own {d['gpu_ms']:.4f}  shared {d['gpu_ms_shared']:.4f}  {d['text'][:60]}")
    c = p.get("crosscheck")
    if c:
        lines += ["", f"  torch.profiler crosscheck: {c['truestar_kernels_seen']} of {c['truestar_kernels_launched']} Truestar "
                      f"kernel launches seen" + (f", {c['dropped']} DROPPED (TS511)" if c["dropped"] else "")
                  + f"; GPU kernel time {c['gpu_ms_per_call']:.3f} ms per call against {t['step_gpu_ms']:.3f} ms from events"]
    return "\n".join(lines)


def _short(f: Optional[str]) -> str:
    if not f:
        return "?"
    return f.replace("\\", "/").rsplit("/", 1)[-1]


def to_agent(p: Dict[str, Any], model_file: Optional[str] = None) -> Dict[str, Any]:
    from ..agent import _envelope
    from ..codes import CODES
    issues = []
    if p["accounting"]["unaccounted"]:
        issues.append({"code": "TS510", "slug": CODES["TS510"][1], "severity": CODES["TS510"][0],
                       "message": f"{len(p['accounting']['unaccounted'])} step runs have no valid GPU time"})
    c = p.get("crosscheck")
    if c and c["dropped"]:
        issues.append({"code": "TS511", "slug": CODES["TS511"][1], "severity": CODES["TS511"][0],
                       "message": f"torch.profiler saw {c['truestar_kernels_seen']} of {c['truestar_kernels_launched']} "
                                  f"kernel launches; the per step events account for all of them"})
    fb = [r for r in p["steps"] if r["kind"] == "fallback"]
    if fb:
        issues.append({"code": "TS512", "slug": CODES["TS512"][1], "severity": CODES["TS512"][0],
                       "message": f"{len(fb)} steps run through the float64 fallback and take "
                                  f"{sum(r['gpu_ms'] for r in fb):.4f} ms per call"})
    status = "error" if p["accounting"]["unaccounted"] else "ok"
    doc = _envelope(status, 0)
    doc["command"] = "profile"
    doc["model"] = {"name": p["program"], "file": model_file, "dtype": p["dtype"]}
    doc["codes"] = list(dict.fromkeys(i["code"] for i in issues))
    doc["issues"] = issues
    doc.update({k: p[k] for k in ("device", "gpu", "iterations", "totals", "accounting", "first_call", "steps",
                                  "lines", "crosscheck", "sentinels")})
    actions = []
    if p["steps"]:
        r = p["steps"][0]
        ln = r["lines"][0] if r["lines"] else None
        actions.append({"kind": "inspect_hot_step", "description":
                        f"{r['step']} ({' + '.join(r['ops'])}) takes {100 * (r['share'] or 0):.1f}% of GPU time",
                        "command": None, "file": ln["file"] if ln else None, "line": ln["line"] if ln else None})
    doc["next_actions"] = actions
    return _clean(doc)


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


# ----------------------------------------------------------------------------- command line


def run_cli(a) -> int:
    import contextlib
    import json
    import os
    import sys

    from ..agent import dumps, error_to_agent
    from ..cli import ModelFileError, _load

    def fail(code, message):
        if a.format == "agent":
            doc = error_to_agent(code, message, a.model)
            doc["command"] = "profile"
            print(dumps(doc))
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        return fail("TS001", "truestar profile times GPU steps and CUDA is not available here")
    quiet = contextlib.redirect_stdout(sys.stderr) if a.format == "agent" else contextlib.nullcontext()
    with quiet:
        try:
            model, inputs, kwargs = _load(a.model)
        except ModelFileError as exc:
            return fail("TS020", str(exc))
        if kwargs:
            return fail("TS021", "profile takes positional tensor inputs only; get_kwargs() is not supported")
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        from .. import ir
        from .executor import compile_program
        try:
            model = model.eval()
            prog = ir.import_module(model, tuple(inputs), name=os.path.splitext(os.path.basename(a.model))[0])
            cp = compile_program(prog, "cuda", a.dtype, allow_tf32=a.tf32, sentinels=a.sentinels)
        except (ValueError, TypeError) as exc:
            return fail("TS021", f"{type(exc).__name__}: {exc}")
        dev = [t.to("cuda") for t in inputs]
        p = profile_program(cp, dev, iters=a.iters, warmup=a.warmup, crosscheck=a.crosscheck)
    doc = to_agent(p, os.path.abspath(a.model))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(dumps(doc))
    if a.format == "agent":
        print(dumps(doc))
    else:
        print(to_text(p, a.top))
    return 1 if doc["status"] == "error" else 0
