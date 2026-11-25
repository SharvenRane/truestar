"""ATen op coverage: every op the model list produces under torch.export, and what Truestar has for it.

    PYTHONPATH=src python scripts/op_coverage.py [--sources tv,hf,nn] [--only resnet18,llama]
        [--compile cpu|cuda] [--train] [--json out.json] [--write-doc]

Sources (random weights, nothing downloaded):

* ``tv``: the torchvision classifier zoo (``torchvision.models.list_models``), up to 100M params, at a
  small input (64, or 224 for ViT, Swin and MaxViT, 299 for Inception v3).
* ``hf``: the ``transformers`` config built models of ``truestar.testing_hf`` (MODELS and
  EXTRA_MODELS: Llama, Mistral, Qwen2, GPT2, BERT, ViT, T5 encoder, Whisper, CLIP text and vision,
  DeiT, Swin, ConvNeXt, MobileViT, Mamba).
* ``nn``: one module per torch.nn layer below, and the common losses.
* ``gen``: the image generation, detection and segmentation models of ``truestar.testing_vision``
  (diffusers UNets, DiT, Flux, the VAE, DDIM and Euler steps; torchvision detectors and segmenters;
  DETR and SAM).
* ``cur``: the ops and attention patterns of the 2025 to 2026 families alone
  (``truestar.testing_ops.current_ops``: diff, floor, bucketize, std, string conv padding, attention
  sinks, logit soft capping, ...).

For each model the default ``torch.export`` graph is read (ops inside control flow regions too), then
imported with ``truestar.ir.import_exported``. Each ATen op is reported with the IR ops its import
produced and, for those IR ops, whether Truestar has a float64 reference, a reverse (gradient) rule,
a forward (pushforward) rule, a batching rule, and what the compiler runs: a native kernel (loop IR,
Triton on CUDA and numba on the CPU), a layout or creation step (no arithmetic), a torch library
step on the device, or the float64 CPU reference (a fallback). ``--compile`` compiles every imported
model and lists the fallbacks that really happened; ``--train`` differentiates the sum of the first
output with respect to every float parameter and checks the gradients against torch.autograd in
float64 (bar 1e-8 relative to the largest gradient).

``--write-doc`` replaces the generated section of docs/MODEL_COVERAGE.md.
"""

from __future__ import annotations

import argparse
import contextlib
import collections
import copy
import io
import json
import os
import re
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch


from truestar import ir
from truestar.ir import grad_rules, reference
from truestar.ir.core import UnsupportedOpError

MAX_PARAMS = 100_000_000
DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "MODEL_COVERAGE.md")
BEGIN, END = "<!-- op_coverage:begin -->", "<!-- op_coverage:end -->"


# ----------------------------------------------------------------------------- the model list


def _tv_models(only):
    import torchvision
    names = torchvision.models.list_models(module=torchvision.models)
    for name in names:
        if only and name not in only:
            continue
        s = 224 if name.startswith(("vit_", "swin", "maxvit")) else 299 if name == "inception_v3" else 64

        def make(name=name, s=s):
            m = torchvision.models.get_model(name, weights=None).eval()
            if sum(p.numel() for p in m.parameters()) > MAX_PARAMS:
                return None
            return m, (torch.randn(1, 3, s, s),)
        yield f"tv/{name}", make


def _hf_models(only):
    try:
        from truestar import testing_hf
    except ImportError:
        return
    names = list(testing_hf.MODELS) + list(testing_hf.EXTRA_MODELS)
    for table in ("MODELS_2026", "MOE_2026", "REFUSED_2026"):    # absent at commits before the coverage refresh
        if hasattr(testing_hf, table):
            names += list(testing_hf.available(getattr(testing_hf, table)))
    for name in names:
        if only and name not in only:
            continue
        yield f"hf/{name}", (lambda name=name: testing_hf.build(name, batch=2))


def _nn_models(only):
    from truestar.testing_ops import nn_layers
    for name, make in nn_layers():
        if only and name not in only:
            continue
        yield f"nn/{name}", make


def _cur_models(only):
    from truestar import testing_ops
    if not hasattr(testing_ops, "current_ops"):     # a commit before the coverage refresh
        return
    for name, make in testing_ops.current_ops():
        if only and name not in only:
            continue
        yield f"op/{name}", make


def _gen_models(only):
    try:
        from truestar import testing_vision
    except ImportError:             # a commit before the vision and image generation coverage
        return
    for name in testing_vision.available(testing_vision.ALL):
        if only and name not in only:
            continue
        yield f"gen/{name}", (lambda name=name: testing_vision.build(name, batch=2))


# ----------------------------------------------------------------------------- what Truestar has per IR op


def _ir_tables():
    from truestar.compiler import lower
    from truestar.ir import forward, vmap
    try:
        from truestar.compiler import coverage_rules
    except ImportError:       # a run at a commit before the op coverage work
        class coverage_rules:  # noqa: N801
            COMPOSITE_OPS = ()
    try:
        from truestar.compiler.gather_native import GATHER_OPS, SCATTER_OPS, ARG_OPS as native_args
        native_reads = GATHER_OPS + SCATTER_OPS
    except ImportError:       # a run at a commit before Truestar's gather kernels
        native_reads, native_args = (), ()
    if not hasattr(lower, "INT_RESULT_OPS"):
        lower.INT_RESULT_OPS = ()
    batching = vmap.batching_table()
    jvp = forward.jvp_rule_table()

    def info(op: str) -> Dict[str, str]:
        if op.endswith("[int]"):         # an integer use of an op (see _ir_ops_by_aten)
            i = info(op[:-5])
            return {**i, "compiler": "integer on device"}
        if op in grad_rules.NONDIFF:
            rev = "none needed"
        elif op in grad_rules.GRAD_RULES:
            rev = "rule"
        elif op in grad_rules.LOWERINGS:
            rev = "lowered"
        elif op in ("cond", "loop"):
            rev = "control flow"
        else:
            rev = "missing"
        fwd = jvp.get(op, "missing")
        bat = batching.get(op, "missing")
        if op in lower.CONTROL_OPS:
            comp = "regions"
        elif op in lower.LAYOUT_OPS:
            comp = "layout"
        elif op in lower.CREATE_OPS or op in lower.HOST_OPS:
            comp = "create"
        elif op in lower.RULES:
            comp = "kernel"
        elif op == "adaptive_max_pool2d" and op in coverage_rules.COMPOSITE_OPS and not native_reads:
            comp = "lowered: torch on device"     # its lowering reads taps with index and gather
        elif op in native_reads or op in native_args:
            comp = "Truestar gather kernel" if op in native_reads else "Truestar argmax kernel"
        elif op in coverage_rules.COMPOSITE_OPS:
            comp = "kernel (lowered)"
        elif op in lower.INT_RESULT_OPS:
            comp = "integer on device"
        elif op in lower.LIBRARY_OPS:
            comp = "library"
        elif op in lower.INDEX_OPS:
            comp = "torch on device"
        elif op in ("random", "random_batched"):
            comp = "kernel"
        else:
            comp = "fallback"
        return {"reference": "yes" if op in reference.IMPLS else "missing", "reverse": rev,
                "forward": "yes" if fwd != "missing" else "missing", "batching": bat if bat != "rule" else "yes",
                "compiler": comp}
    return info


# ----------------------------------------------------------------------------- measuring one model


def _target_name(node) -> str:
    t = node.target
    return str(t) if not hasattr(t, "name") or not callable(getattr(t, "name", None)) else f"aten.{t.name()}"


def _graph_ops(gm) -> collections.Counter:
    """ATen ops of a graph module and of every subgraph it holds (control flow regions)."""
    out = collections.Counter()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        if name.startswith("<built-in") or name.startswith("<function"):
            if "getitem" in name or getattr(node.target, "__module__", "") == "_operator":
                continue
            name = getattr(node.target, "__name__", name)
        out[name] += 1
    for _, sub in gm.named_children():
        if isinstance(sub, torch.fx.GraphModule):
            out.update(_graph_ops(sub))
    return out


def _source_op(src: str) -> str:
    return src.split(" ", 1)[0] if src else ""


def _ir_ops_by_aten(prog) -> Dict[str, collections.Counter]:
    out: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    try:
        from truestar.compiler.lower import INT_DEVICE_OPS
    except ImportError:
        INT_DEVICE_OPS = ()

    def walk(nodes):
        for n in nodes:
            # integer arithmetic (positions, packed sequence ids) runs exactly on the device
            dt = getattr(getattr(n.output, "type", None), "dtype", None)
            integer = n.op in INT_DEVICE_OPS and dt is not None and not dt.is_floating_point
            out[_source_op(n.source)][n.op + ("[int]" if integer else "")] += 1
            for r in (n.attrs.get("regions") or ()):
                walk(getattr(r, "nodes", ()))
            for k in ("body", "cond", "then_branch", "else_branch"):
                r = n.attrs.get(k)
                if r is not None and hasattr(r, "nodes"):
                    walk(r.nodes)
    walk(prog.nodes)
    return out


def _unsupported_ops(msg: str) -> List[str]:
    ops = []
    for ln in msg.splitlines():
        m = re.search(r"unsupported op (\S+)", ln)
        if m:
            ops.append(m.group(1))
    return ops


def _to64(args):
    return tuple(a.double() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)


def _train_check(model64, args64, prog) -> str:
    """Gradient of the sum of output 0 with respect to every float param (or, for a module without
    params, every float input): Truestar's program against torch.autograd, both in float64."""
    from truestar.ir.autodiff import grad
    # trainable weights only (Whisper's positions are frozen); a tied weight is one tensor under one name
    sd = model64.state_dict(keep_vars=True)
    params = {v.name: sd[v.name] for v in prog.params
              if v.name in sd and isinstance(sd[v.name], torch.nn.Parameter) and sd[v.name].requires_grad}
    names = list(params)
    on_inputs = not names
    if on_inputs:
        names = [v.name for v, a in zip(prog.inputs, args64) if torch.is_tensor(a) and a.is_floating_point()]
    if not names:
        return "nothing to differentiate"
    # a random cotangent, not ones: the sum of a softmax (Softmin, Softmax2d) is constant
    step = grad(prog, wrt=names, of=[0], seed="input")
    out0 = ir.run(prog, *args64)
    out0 = out0[0] if isinstance(out0, (list, tuple)) else out0
    cot = torch.randn(out0.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    got = ir.run(step, *args64, cot)
    got = got if isinstance(got, (list, tuple)) else [got]
    g = dict(zip(names, got[len(prog.outputs):]))
    model64.zero_grad(set_to_none=True)
    leaves = [a.clone().requires_grad_(True) if torch.is_tensor(a) and a.is_floating_point() else a for a in args64]
    with ir.reference.exact_float32_casts():
        out = model64(*leaves)
    out = out[0] if isinstance(out, (list, tuple)) else out
    out.backward(cot)
    if on_inputs:
        wants = [a.grad for a in leaves if torch.is_tensor(a) and a.requires_grad]
    else:
        wants = [params[nm].grad for nm in names]
    worst, scale = 0.0, 0.0
    for nm, want in zip(names, wants):
        want = torch.zeros_like(g[nm]) if want is None else want
        worst = max(worst, (g[nm].double() - want.double()).abs().max().item())
        scale = max(scale, want.abs().max().item())
    rel = worst / (scale or 1.0)
    return f"PASS {rel:.1e}" if rel < 1e-8 else f"FAIL {rel:.1e}"


def measure(name: str, make, compile_dev: Optional[str], train: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {"name": name}
    t0 = time.perf_counter()
    try:
        built = make()
    except Exception as e:  # noqa: BLE001
        res["status"] = f"cannot build: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        return res
    if built is None:
        res["status"] = "skipped (over 100M params)"
        return res
    model, args = built
    model = model.eval()
    try:      # as ir.import_module: covered MoE experts export in their static form (truestar.moe)
        from truestar.moe.coverage import lowered
    except ImportError:          # a run at a commit before MoE support
        lowered = None
    # only around the refusal check, the export and the import: the float64 comparison below runs
    # the model's own eager experts (a deep copy of a lowered model would carry the static form)
    low = contextlib.ExitStack()
    if lowered is not None:
        low.enter_context(lowered(model))
    try:
        return _measure(res, name, model, args, compile_dev, train, t0, low)
    finally:
        low.close()


def _measure(res, name, model, args, compile_dev, train, t0, low):
    try:      # as ir.import_module: a family refused by name is refused before torch.export
        from truestar.ir.refusals import check_module
        check_module(model, name)
    except ImportError:          # a run at a commit before the coverage refresh
        pass
    except UnsupportedOpError as e:
        res["import"], res["refused"] = "refused", str(e).splitlines()[0][:300]
        res["aten"] = {}
        res["seconds"] = round(time.perf_counter() - t0, 1)
        return res
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = torch.export.export(model, tuple(args), strict=False)
    except Exception as e:  # noqa: BLE001
        res["status"] = f"torch.export failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        return res
    res["aten"] = dict(_graph_ops(ep.graph_module))
    try:
        prog = ir.import_exported(ep, name=name)
        res["import"] = "PASS"
        res["ir_by_aten"] = {k: dict(v) for k, v in _ir_ops_by_aten(prog).items()}
    except UnsupportedOpError as e:
        if str(getattr(e, "code", "")) in ("TS940", "TS941", "TS942", "TS1260"):   # refused by name (ir/refusals.py)
            res["import"] = "refused"
            res["refused"] = str(e).splitlines()[0][:300]
        else:
            res["import"] = "unsupported"
            res["unsupported"] = sorted(set(_unsupported_ops(str(e)))) or [str(e).splitlines()[0][:150]]
        prog = None
    except Exception as e:  # noqa: BLE001
        res["import"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
        prog = None
    low.close()
    if prog is not None:
        # float64 agreement of the import (the importer's own check is shape only)
        try:
            m64 = copy.deepcopy(model).double()
            a64 = _to64(args)
            p64 = ir.import_module(m64, a64)
            with torch.no_grad(), ir.reference.exact_float32_casts():
                want = m64(*a64)
            want = want if isinstance(want, (list, tuple)) else [want]
            got = ir.run(p64, *a64)
            got = got if isinstance(got, (list, tuple)) else [got]
            err = max(((g.double() - w.double()).abs().max() / (w.double().abs().max() or 1.0)).item()
                      for g, w in zip(got, want) if torch.is_tensor(w) and w.numel())
            res["f64_error"] = err
            if train:
                try:
                    res["train"] = _train_check(m64, a64, p64)
                except UnsupportedOpError as e:
                    ops = sorted(set(re.findall(r"no gradient rule for IR op '(\w+)'", str(e))))
                    res["train"] = ("no gradient rule: " + " ".join(ops)) if ops else f"error: {str(e)[:150]}"
                except Exception as e:  # noqa: BLE001
                    res["train"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
        except Exception as e:  # noqa: BLE001
            res["f64_error"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
        if compile_dev:
            try:
                from truestar.compiler import compile_program
                cp = compile_program(prog, compile_dev, "float32")
                dev_args = [a.to(compile_dev) if torch.is_tensor(a) else a for a in args]
                cp(dev_args)
                fb = cp.report.fallbacks
                res["compile"] = "PASS"
                res["fallback_ops"] = sorted({o for f in fb for o in f["ops"]})
            except Exception as e:  # noqa: BLE001
                res["compile"] = f"error: {type(e).__name__}: {str(e).splitlines()[0][:150]}"
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


# ----------------------------------------------------------------------------- the table


def table(results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    info = _ir_tables()
    supported = set(ir.supported_aten_ops())
    per_op: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r.get("import") == "refused":    # a refused family's ops are not claimed either way
            continue
        for op, n in (r.get("aten") or {}).items():
            e = per_op.setdefault(op, {"models": 0, "uses": 0, "ir": collections.Counter(), "names": []})
            e["models"] += 1
            e["uses"] += n
            e["names"].append(r["name"])
        for op, irs in (r.get("ir_by_aten") or {}).items():
            if op in per_op:
                per_op[op]["ir"].update(irs)
    rows = []
    summary = collections.Counter()
    for op, e in sorted(per_op.items(), key=lambda kv: (-kv[1]["models"], -kv[1]["uses"], kv[0])):
        imported = op in supported or op in ("cond", "while_loop", "scan", "wrap_with_set_grad_enabled",
                                             "wrap_with_autocast", "sym_size", "sym_numel")
        irs = [o for o, _ in e["ir"].most_common()]
        cols = {"reference": [], "reverse": [], "forward": [], "batching": [], "compiler": []}
        for o in irs:
            i = info(o)
            for k in cols:
                cols[k].append(i[k])

        def worst(vals, good):
            if not imported:
                return "no import"
            if not vals:
                return "n/a"
            bad = sorted({v for v in vals if v not in good})
            return ", ".join(bad) if bad else "yes"
        comp = sorted(set(cols["compiler"]))
        row = {
            "aten": op, "models": e["models"], "uses": e["uses"], "import": "yes" if imported else "NO",
            "ir": " ".join(irs[:6]) + (" ..." if len(irs) > 6 else ""),
            "reference": worst(cols["reference"], {"yes"}),
            "reverse": worst(cols["reverse"], {"rule", "lowered", "none needed", "control flow"}),
            "forward": worst(cols["forward"], {"yes"}),
            "batching": worst(cols["batching"], {"yes", "random", "no operands", "loop"}),
            "compiler": (", ".join(comp) if comp else "n/a") if imported else "no import",
        }
        rows.append(row)
        summary["ops"] += 1
        summary["imported"] += imported
        for k in ("reference", "reverse", "forward", "batching"):
            summary[k] += row[k] == "yes" or row[k] == "n/a"
        # an op that imports to no IR op at all (dropout in eval, contiguous, alias) costs nothing
        how = " ".join(comp)
        summary["native"] += imported and not ("fallback" in how or "library" in how or "on device" in how)
        summary["device"] += imported and "on device" in how
        summary["fallback"] += imported and "fallback" in how
    lines = ["| ATen op | models | uses | import | IR ops | f64 ref | reverse | forward | batching | compiled as |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['aten']}` | {r['models']} | {r['uses']} | {r['import']} | {r['ir']} | {r['reference']} | "
                     f"{r['reverse']} | {r['forward']} | {r['batching']} | {r['compiler']} |")
    return "\n".join(lines), {"rows": rows, "summary": dict(summary)}


def model_table(results) -> str:
    lines = ["| model | ATen ops | import | f64 error | compile fallbacks | train step (grad vs torch) |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if "aten" not in r:
            lines.append(f"| {r['name']} | | {r.get('status')} | | | |")
            continue
        imp = r.get("import")
        if imp == "unsupported":
            imp = "NO: " + ", ".join(f"`{o}`" for o in r["unsupported"])
        elif imp == "refused":
            m = re.match(r"\[(TS9\d\d)\].*?has (.+?) \((\w+) at", r["refused"])
            imp = f"refused {m.group(1)}: {m.group(2)} (`{m.group(3)}`)" if m else "refused: " + r["refused"][:80]
        err = r.get("f64_error")
        err = f"{err:.1e}" if isinstance(err, float) else (err or "")
        comp = r.get("compile")
        if comp == "PASS":
            comp = ", ".join(r["fallback_ops"]) or "none"
        lines.append(f"| {r['name']} | {len(r['aten'])} | {imp} | {err} | {comp or ''} | {r.get('train', '')} |")
    return "\n".join(lines)


def _changes(before_rows, after_rows) -> str:
    """Ops whose import, rules or compiled path differ between two runs (each as its run measured it)."""
    old = {r["aten"]: r for r in before_rows}
    keys = ("import", "reference", "reverse", "forward", "batching", "compiler")
    lines = ["| ATen op | models | before: import, reverse, forward, compiled as | after |", "|---|---|---|---|"]
    for r in after_rows:
        b = old.get(r["aten"])
        if b is None or any(b[k] != r[k] for k in keys):
            was = "not in the list" if b is None else f"{b['import']}, {b['reverse']}, {b['forward']}, {b['compiler']}"
            lines.append(f"| `{r['aten']}` | {r['models']} | {was} | {r['import']}, {r['reverse']}, {r['forward']}, "
                         f"{r['compiler']} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="tv,hf,nn,cur")
    p.add_argument("--only", default="")
    p.add_argument("--compile", default="", help="cpu or cuda: compile every imported model")
    p.add_argument("--train", action="store_true")
    p.add_argument("--json", default="")
    p.add_argument("--from-json", default="", help="rebuild the tables from saved runs (comma separated)")
    p.add_argument("--write-doc", action="store_true")
    p.add_argument("--before", default="", help="a saved run of an earlier commit: the doc gets a table of "
                                                "the ops whose status changed")
    a = p.parse_args(argv)
    only = set(x for x in a.only.split(",") if x)
    if a.from_json:
        results = []
        for path in a.from_json.split(","):
            results += json.load(open(path))["results"]
    else:
        gens = {"tv": _tv_models, "hf": _hf_models, "nn": _nn_models, "cur": _cur_models, "gen": _gen_models}
        results = []
        for src in a.sources.split(","):
            for name, make in gens[src](only):
                torch.manual_seed(0)
                quiet = io.StringIO()
                old = sys.stdout
                sys.stdout = quiet
                try:
                    r = measure(name, make, a.compile or None, a.train)
                except Exception:  # noqa: BLE001
                    r = {"name": name, "status": "script error: " + traceback.format_exc().splitlines()[-1][:150]}
                finally:
                    sys.stdout = old
                results.append(r)
                err = r.get("f64_error")
                err = f"{err:.1e}" if isinstance(err, float) else (err or "")
                print(f"{name:40s} {r.get('import', r.get('status'))} {err} {r.get('unsupported', '')} "
                      f"{r.get('compile', '')} {r.get('fallback_ops', '')} {r.get('train', '')} {r.get('seconds', '')}s",
                      flush=True)
    ops_md, data = table(results)
    models_md = model_table(results)
    s = data["summary"]
    n_models = sum(1 for r in results if "aten" in r)
    n_import = sum(1 for r in results if r.get("import") == "PASS")
    n_refused = sum(1 for r in results if r.get("import") == "refused")
    refused = f", {n_refused} refused by name (TS940 to TS942, TS1260)" if n_refused else ""
    head = (f"{n_models} models exported, {n_import} import{refused}. {s.get('ops', 0)} distinct ATen ops: "
            f"{s.get('imported', 0)} import; of all ops, {s.get('reference', 0)} have a float64 reference for every "
            f"IR op they produce, {s.get('reverse', 0)} a reverse rule, {s.get('forward', 0)} a forward rule, "
            f"{s.get('batching', 0)} a batching rule, {s.get('native', 0)} compile to native steps only "
            "(kernels, layout or creation, or nothing at all), "
            f"{s.get('device', 0)} use a torch step on the device for part of their work, "
            f"{s.get('fallback', 0)} can take the float64 CPU fallback.")
    print("\n" + head)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        # the tables as measured by this code (a run at another commit keeps its own answers)
        json.dump({"results": results, "summary": s, "head": head, "rows": data["rows"], "ops_md": ops_md,
                   "models_md": models_md}, open(a.json, "w"), indent=1, default=str)
    if a.write_doc:
        parts = [BEGIN, "", head, ""]
        if a.before:
            parts += ["### Changed since the earlier run", "", _changes(json.load(open(a.before))["rows"], data["rows"]),
                      ""]
        parts += ["### Per ATen op", "", ops_md, "", "### Per model", "", models_md, "", END]
        body = "\n".join(parts)
        doc = open(DOC, encoding="utf-8").read()
        if BEGIN in doc:
            doc = doc[:doc.index(BEGIN)] + body + doc[doc.index(END) + len(END):]
        else:
            doc = doc.rstrip() + "\n\n## ATen op coverage (generated)\n\n" + body + "\n"
        open(DOC, "w", encoding="utf-8").write(doc)
    return results


if __name__ == "__main__":
    main()
