"""`truestar check` on a compiler or runtime backend (torch.compile Inductor, ONNX Runtime, TensorRT).

Design (docs/BACKENDS.md)
-------------------------
1. Verdict from the real engine. The whole model is compiled the way a user compiles it (default
   flags, graph outputs only) and its output is compared with the float64 reference. Nothing is
   added to that engine to observe it: marking intermediate tensors as outputs changes what a
   compiler may fuse, so the engine being judged would no longer be the one that ships. On a match
   the check stops here; its cost is one compile and one run on top of the reference.
2. Localization only after a mismatch, in separate engines. Descent over the module tree: each
   child of the current module, in execution order, is compiled alone by the same stack and fed
   its float64 reference inputs (cast to the model dtype), so its error is its own. The first child
   over threshold becomes the current module and the descent repeats; it stops at a module whose
   children all match alone. That module is blamed. If it is the whole model while it has
   children, the fault needs modules compiled together (TS402).
   A transformer with 12 identical blocks whose 4th is bad costs 4 block compiles at the top level
   plus the children of block 4, not one compile per module.
3. Op drilldown in separate engines: each ATen op of the blamed module (traced in the float64
   reference) is compiled alone by the stack with its reference inputs; the first one over
   threshold is named, and the same op run eagerly on the device tells a compiler fault from a
   device kernel fault. For Inductor the named op is traced to the fused kernel that computes it in
   the blamed module's engine (Inductor records the ATen origins of every kernel).
4. Thresholds use the precision the stack computes in (float16 for trt:fp16) with the same
   calibration as the eager check: never below 10x the error a correct CPU run in that precision
   shows on the same module or op. TF32 tolerance applies only when the stack runs TF32 (Inductor
   with torch's TF32 flags, ORT CUDA's use_tf32, trt:tf32) and only to calls with a matmul or conv.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .capture import ROOT, flatten_tensors, is_tf32_op, map_tensors
from .check import (BackendResult, LayerResult, _cast_inputs, _decisions, _module_by_name,
                    NoiseFloor)
from .compare import (DIVERGENT, DIVERGENT_FACTOR, DTYPE_FLOOR, OK, SUSPECT, diff_lists,
                      threshold_for, worst)
from .stacks import StackBackend, StackError

MAX_OP_COMPILES = 48  # op drilldown compiles at most this many single op engines


def _sig(obj: Any) -> tuple:
    return tuple((tuple(t.shape), str(t.dtype)) for t in flatten_tensors(obj))


def _aliasing(func) -> bool:
    """Views and in place ops: nothing to compute, or nothing a functional graph can return."""
    try:
        return any(r.alias_info is not None for r in func._schema.returns)
    except Exception:
        return False


def _usable_noise(n: Optional[float], eff: torch.dtype, dtype: torch.dtype) -> Optional[float]:
    """A calibrated noise floor, or None when it cannot be trusted. When the stack computes in a
    lower precision than the model (trt:fp16 on float32), a CPU run in that precision that is off
    by more than the precision's own floor is not rounding: it hit the same range limit (float16
    overflow) the stack may hit, and using it would excuse exactly the fault to report."""
    if n is None or n != n:
        return None
    limit = DTYPE_FLOOR.get(eff, 1e-4) * (1.0 if eff != dtype else DIVERGENT_FACTOR)
    return n if n <= limit else None


def _whole_noise(base: nn.Module, args, kwargs, ref_flat, dtype) -> Optional[float]:
    """Error of the whole model run on CPU in `dtype`, for a stack that computes in a lower
    precision than the model (trt:fp16 on a float32 model)."""
    try:
        m = copy.deepcopy(base).to(dtype)
        with torch.no_grad():
            out = m(*_cast_inputs(args, "cpu", dtype), **_cast_inputs(kwargs, "cpu", dtype))
        e = diff_lists([t.detach().cpu() for t in flatten_tensors(out)], ref_flat).rel_err
        return e if math.isfinite(e) else None
    except Exception:
        return None


def check_stack(res: BackendResult, base: nn.Module, args, kwargs, dtype: torch.dtype, ref_calls,
                ref_flat, noise: NoiseFloor, op_counting: bool, repro_dir: Optional[str],
                model_name: str, ref_dtype: Optional[torch.dtype],
                options: Optional[Dict[str, Any]] = None) -> None:
    b: StackBackend = res.backend  # type: ignore[assignment]
    if options:
        b.options = dict(options)
        res.notes.append("stack options were set (" + ", ".join(sorted(options)) + "): this is "
                         "a modified stack, and the repro does not carry the modification.")
    eff = b.compute_dtype(dtype)
    runner = b.make_runner()
    info: Dict[str, Any] = {
        "kind": b.kind, "option": b.option,
        "compute_dtype": str(eff).replace("torch.", ""),
        "engine": "whole model compiled as users compile it; no extra outputs",
        "engine_seconds": None, "localization": "not needed", "localize_seconds": 0.0,
        "compiles": 0, "op_compiles": 0,
    }
    res.stack = info
    engines: Dict[tuple, Any] = {}
    try:
        _run(res, b, runner, info, engines, base, args, kwargs, dtype, eff, ref_calls, ref_flat,
             noise, op_counting, repro_dir, model_name, ref_dtype)
    finally:
        for e in engines.values():
            try:
                e.close()
            except Exception:
                pass
        info["compiles"] = runner.compiles
        runner.close()


def _run(res, b, runner, info, engines, base, args, kwargs, dtype, eff, ref_calls, ref_flat,
         noise, op_counting, repro_dir, model_name, ref_dtype) -> None:
    by_key = {c.key: c for c in ref_calls}
    root = next(c for c in reversed(ref_calls) if c.parent is None)

    # 1. The real engine: the whole model, compiled once, judged by its output.
    t0 = time.perf_counter()
    a_in, k_in = _cast_inputs(args, "cpu", dtype), _cast_inputs(kwargs, "cpu", dtype)
    whole = runner.compile(base, a_in, k_in, whole_model=True)
    engines[("whole",)] = whole
    out = whole(a_in, k_in)
    info["engine_seconds"] = round(time.perf_counter() - t0, 3)
    if "tf32" in whole.info:
        b.tf32 = bool(whole.info["tf32"])
    if "providers" in whole.info:
        info["providers"] = whole.info["providers"]
    for code, msg in whole.findings:
        res.add(code, msg)
        res.status = worst(res.status, SUSPECT)
    gb = whole.info.get("graph_breaks") or 0
    if gb:
        info["graph_breaks"] = gb
        res.add("TS403", f"torch.compile split the model at {gb} graph break(s); the code between "
                         "graphs ran eagerly, as it does for the user")
    if whole.info.get("kernels"):
        info["kernels"] = len(whole.info["kernels"])

    res.output = diff_lists(out, ref_flat)
    model_tf32 = b.tf32 and (not op_counting or any(c.tf32_ops > 0 for c in ref_calls))
    out_noise = _whole_noise(base, args, kwargs, ref_flat, eff) if eff != dtype else None
    thr_out = threshold_for(eff, model_tf32, _usable_noise(out_noise, eff, dtype))
    out_status = res.output.status(thr_out)
    res.status = worst(res.status, out_status)
    if out_flat_ok(out, ref_flat):
        n, flips, ties = _decisions(out[0], ref_flat[0], thr_out)
        res.decisions, res.flips, res.tie_flips = n, flips, ties
        if flips - ties > 0:
            res.status = worst(res.status, SUSPECT)
            res.add("TS013", f"{flips - ties} of {n} predictions changed beyond near ties "
                             f"({ties} near ties)")
    root_layer = LayerResult(root.key, root.name, root.type_name, root.is_leaf, res.output, None,
                             thr_out, out_status, True, "")
    if out_status == OK and res.flips - res.tie_flips <= 0:
        res.layers = [root_layer]
        info["localization"] = "not needed: the whole model engine matches the reference"
        return

    # 2. Localization by descent over the module tree, one separate engine per module.
    t1 = time.perf_counter()
    tested: List[LayerResult] = []
    infos: Dict[str, Dict[str, Any]] = {root.key: whole.info}

    def isolate(c) -> LayerResult:
        a, k = _cast_inputs(c.args, "cpu", dtype), _cast_inputs(c.kwargs, "cpu", dtype)
        n = _usable_noise(noise.get(c.key, eff, always=True), eff, dtype)
        thr = threshold_for(eff, b.tf32 and (not op_counting or c.tf32_ops > 0), n)
        sig = (c.name,) + _sig((a, k))
        try:
            eng = engines.get(sig)
            if eng is None:
                eng = runner.compile(_module_by_name(base, c.name), a, k)
                engines[sig] = eng
            outs = eng(a, k)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            return LayerResult(c.key, c.name, c.type_name, c.is_leaf, None, None, thr, OK, False,
                               msg)
        infos[c.key] = eng.info
        iso = diff_lists(outs, c.outputs)
        return LayerResult(c.key, c.name, c.type_name, c.is_leaf, iso, None, thr, iso.status(thr),
                           True, "")

    node = root
    unbuilt: List[str] = []
    while by_key[node.key].children:
        pick = suspect = None
        for ck in sorted(by_key[node.key].children, key=lambda k: by_key[k].start):
            lr = isolate(by_key[ck])
            tested.append(lr)
            if not lr.replayed:
                unbuilt.append(f"{lr.key} ({lr.error[:160]})")
            if lr.status == DIVERGENT:
                pick = by_key[ck]
                break
            if lr.status == SUSPECT and suspect is None:
                suspect = by_key[ck]
        pick = pick or suspect
        if pick is None:
            break
        node = pick
    info["localize_seconds"] = round(time.perf_counter() - t1, 3)
    info["localization"] = (f"descent over the module tree: {len(tested)} module(s) compiled "
                            "alone with float64 reference inputs, after the whole model engine "
                            "mismatched")
    if unbuilt:
        res.notes.append(f"{len(unbuilt)} module(s) could not be compiled alone by {b.name} and "
                         "were skipped: " + "; ".join(unbuilt[:4]))

    res.layers = sorted(tested + [root_layer], key=lambda l: by_key[l.key].start)
    blamed = root_layer if node is root else next(l for l in tested if l.key == node.key)
    res.first_divergent = blamed
    fd = blamed
    res.status = worst(res.status, fd.status)
    own = fd.isolated.rel_err if fd.isolated is not None else float("nan")
    if node is root and not root.is_leaf:
        res.add("TS402", f"the whole model compiled by {b.name} has error {own:.2e} (threshold "
                         f"{fd.threshold:.1e}) but each of its {len(by_key[root.key].children)} "
                         "top level modules compiled alone matches: the fault needs them compiled "
                         "together (fusion across modules) or is in code between them")
    else:
        res.add("TS010" if fd.status == DIVERGENT else "TS011",
                f"module {fd.key} ({fd.type_name}) compiled alone by {b.name}: own error "
                f"{own:.2e} vs threshold {fd.threshold:.1e}")
    try:
        from .agent import module_source

        res.blame_source = module_source(base, fd.name)
    except Exception:
        res.blame_source = None

    binfo = infos.get(fd.key, {})
    if b.kind == "trt" and callable(binfo.get("layer_names")):
        names = binfo["layer_names"]()
        if names:
            info["blamed_engine_layers"] = names[:32]
    if b.kind == "inductor" and binfo.get("kernels"):
        info["blamed_kernels"] = binfo["kernels"][:32]

    # 3. Op drilldown in separate single op engines.
    c = by_key[fd.key]
    if ref_dtype is not None and fd.replayed:
        t2 = time.perf_counter()
        try:
            _drill(res, b, runner, info, base, c, fd, dtype, eff, ref_dtype, binfo)
        except Exception as exc:
            res.notes.append(f"op drilldown failed on {c.key}: {type(exc).__name__}: {exc}")
        info["drill_seconds"] = round(time.perf_counter() - t2, 3)

    if repro_dir:
        from .stackrepro import write_stack_op_repro, write_stack_repro

        try:
            res.repro_path = write_stack_repro(repro_dir, model_name, b, c,
                                               _module_by_name(base, c.name), dtype, fd.threshold)
        except Exception as exc:
            res.notes.append(f"No repro written ({type(exc).__name__}: {exc}).")
        if res.first_divergent_op is not None:
            try:
                res.op_repro_path = write_stack_op_repro(repro_dir, model_name, b, c,
                                                         res.first_divergent_op)
            except Exception as exc:
                res.notes.append(f"No op repro written ({type(exc).__name__}: {exc}).")


def out_flat_ok(out, ref_flat) -> bool:
    return bool(out) and bool(ref_flat)


def _cast_floats(obj: Any, dtype: torch.dtype) -> Any:
    return map_tensors(obj, lambda t: t.detach().to(dtype) if t.is_floating_point() else t)


def _eager(func, args, kwargs, device: str) -> List[torch.Tensor]:
    from .opdrill import _call

    return _call(func, args, kwargs, device)


def _drill(res, b, runner, info, base, c, fd, dtype, eff, ref_dtype, binfo) -> None:
    from .opdrill import MAX_OPS, OpResult, _comparable, _leaves, trace_ops

    ref_mod = copy.deepcopy(_module_by_name(base, c.name)).to(ref_dtype)
    clone = lambda o: map_tensors(o, torch.clone)
    ops, truncated = trace_ops(ref_mod, clone(c.args), clone(c.kwargs), MAX_OPS)
    if truncated:
        res.notes.append(f"op drilldown skipped for {c.key}: more than {MAX_OPS} ops.")
        return
    ops = [r for r in ops if _comparable(r) and not _aliasing(r.func)]
    total = len(ops)
    found: Optional[OpResult] = None
    tried = failed = 0
    first_error = ""
    for i, r in enumerate(ops):
        if tried >= MAX_OP_COMPILES:
            res.notes.append(f"op drilldown on {c.key} stopped after {MAX_OP_COMPILES} single op "
                             f"compiles ({total} ops in the module).")
            break
        a, k = _cast_floats(r.args, dtype), _cast_floats(r.kwargs, dtype)
        tried += 1
        try:
            eng = runner.compile_op(r.func, a, k)
            try:
                got = eng(a, k)
            finally:
                eng.close()
        except Exception as exc:
            failed += 1
            first_error = first_error or f"{r.name}: {type(exc).__name__}: {str(exc)[:200]}"
            continue
        want = r.outputs[:1] if eng.info.get("first_output_only") else r.outputs
        iso = diff_lists(got, want)
        thr = fd.threshold
        try:
            cpu = _eager(r.func, _cast_floats(r.args, eff), _cast_floats(r.kwargs, eff), "cpu")
            n = _usable_noise(diff_lists(cpu[:len(want)], want).rel_err, eff, dtype)
            thr = max(thr, threshold_for(eff, b.tf32 and is_tf32_op(r.func), n))
        except Exception:
            pass
        status = iso.status(thr)
        if status == OK:
            continue
        leaves = _leaves((a, k))
        op = OpResult(op=r.name, module=c.key, index=i + 1, total=total,
                      input_shapes=[list(t.shape) for t in leaves],
                      input_dtypes=[str(t.dtype).replace("torch.", "") for t in leaves],
                      isolated=iso, accumulated=None, threshold=thr, status=status, matched=True,
                      args=a, kwargs=k, reference=list(want))
        if found is None or (found.status != DIVERGENT and status == DIVERGENT):
            found = op
        if status == DIVERGENT:
            break
    info["op_compiles"] = tried
    if failed:
        res.notes.append(f"op drilldown on {c.key}: {failed} of {tried} ops could not be compiled "
                         f"alone by {b.name} and were skipped (first: {first_error}).")
    if found is None:
        msg = (f"op drilldown on {c.key}: every ATen op compiled alone by {b.name} matches, so "
               "the fault appears only when the ops are compiled together (fusion)")
        if binfo.get("kernels"):
            ks = [f"{k['kernel']} ({', '.join(k['aten'])})" for k in binfo["kernels"][:6]]
            msg += ". Its kernels: " + "; ".join(ks)
        res.notes.append(msg + ".")
        return
    res.first_divergent_op = found
    # Same op, eager, on the stack's device: separates a compiler fault from a device kernel fault.
    try:
        dev = "cuda:0" if b.device.startswith("cuda") else "cpu"
        prec = str(eff).replace("torch.", "")
        e = diff_lists(_eager(_func_of(found, ops), _cast_floats(found.args, eff),
                              _cast_floats(found.kwargs, eff), dev)[:len(found.reference)],
                       found.reference)
        if e.status(found.threshold) == OK:
            res.notes.append(f"{found.op} run eagerly on {dev} in {prec} matches the reference "
                             f"(error {e.rel_err:.1e}), so the fault is in {b.name}'s code for "
                             "this op, not in the device kernel or the precision.")
        elif eff != dtype:
            res.notes.append(f"{found.op} run eagerly on {dev} in {prec} is wrong too (error "
                             f"{e.rel_err:.1e}{', NaN/Inf' if e.nonfinite_mismatch else ''}): "
                             f"{prec} itself cannot hold this op's values; keep it in "
                             f"{str(dtype).replace('torch.', '')}.")
        else:
            res.notes.append(f"{found.op} run eagerly on {dev} is wrong too (error "
                             f"{e.rel_err:.1e}): the device kernel itself is at fault.")
    except Exception:
        pass
    if b.kind == "inductor" and binfo.get("kernels"):
        packet = found.op.rsplit(".", 1)[0]
        hits = [k for k in binfo["kernels"] if packet in k["aten"]]
        if hits:
            k = hits[0]
            found.kernel = {"kernel": k["kernel"], "aten": k["aten"], "extern": k["extern"]}
            res.notes.append(f"in the compiled {c.key}, {found.op} is computed by "
                             f"{'library call' if k['extern'] else 'fused kernel'} {k['kernel']} "
                             f"(fused ATen ops: {', '.join(k['aten'])}).")


def _func_of(op, ops):
    for r in ops:
        if r.name == op.op:
            return r.func
    raise LookupError(op.op)
