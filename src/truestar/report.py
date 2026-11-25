"""Plain text report."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .compare import DIVERGENT, OK, SUSPECT

if TYPE_CHECKING:
    from .check import CheckReport

LABEL = {OK: "OK", SUSPECT: "SUSPECT", DIVERGENT: "DIVERGENT"}


def _e(x) -> str:
    if x is None:
        return "-"
    if x == float("inf"):
        return "inf"
    return f"{x:.1e}"


def render_text(report: "CheckReport", top: int = 5) -> str:
    lines = []
    dt = str(report.dtype).replace("torch.", "")
    lines.append(f"truestar check: {report.model_name} ({dt}), reference {report.reference}")
    for n in report.notes:
        lines.append(f"  note: {n}")
    for i in report.issues:
        lines.append(f"  {i.code}: {i.message}")
    for name, reason in report.skipped:
        lines.append(f"  skipped {name} (TS001): {reason}")
    lines.append("")
    for r in report.results:
        b = r.backend
        codes = f"  {' '.join(r.codes)}" if r.codes else ""
        lines.append(f"[{LABEL[r.status]}] {b.name}: {b.description}{codes}")
        if r.error:
            lines.append(f"  TS002 failed to run: {r.error}")
        for i in r.issues:
            if i.code in ("TS004", "TS012", "TS961") or i.code.startswith("TS56"):  # TS56x: WebGPU
                lines.append(f"  {i.code}: {i.message}")
        for n in r.notes:
            for i, part in enumerate(n.rstrip().splitlines()):
                lines.append(("  note: " if i == 0 else "        ") + part)
        if r.precision is not None:
            from .precision_probe import manifest_line

            lines.append("  " + manifest_line(r.precision))
        if r.output is not None:
            lines.append(f"  final output: relative error {_e(r.output.rel_err)}"
                         + (f", worst slice {_e(r.output.local_err)} at {r.output.local_at}"
                            if r.output.local_at else "")
                         + ("  (NaN/Inf mismatch)" if r.output.nonfinite_mismatch else ""))
        if r.decisions:
            real = r.flips - r.tie_flips
            tag = "TS013 " if "TS013" in r.codes else ""
            lines.append(f"  {tag}decisions: {r.flips}/{r.decisions} predictions changed"
                         + (f" ({r.tie_flips} near ties, {real} real)" if r.flips else ""))
        fd = r.first_divergent
        if fd is not None:
            iso = fd.isolated
            why = fd.error or (iso.note if iso and iso.note else "")
            if iso is not None and iso.nonfinite_mismatch:
                why = "NaN/Inf where the reference has none"
            where = "" if fd.is_leaf else " in its own forward code, outside its submodules"
            code = "TS010" if fd.status == DIVERGENT else "TS011"
            kind = "divergent" if fd.status == DIVERGENT else "suspect"
            if r.stack is not None:
                where = " compiled alone by this backend"
                if "TS402" in r.codes:
                    code, kind, where = "TS402", "divergent only when compiled together:", ""
            lines.append(
                f"  {code} first {kind} module: {fd.key} ({fd.type_name}){where}, own "
                + (iso.explain(fd.threshold) if iso is not None and iso.localized(fd.threshold)
                   else f"error {_e(iso.rel_err if iso else None)} vs threshold {_e(fd.threshold)}")
                + (f", {why}" if why else "")
            )
            src = (r.blame_source or {}).get("user")
            if src is not None:
                lines.append(f"  source: {src['file']}:{src['line']} ({src['qualname']})")
        op = r.first_divergent_op
        if op is not None:
            why = "  (NaN/Inf mismatch)" if op.isolated.nonfinite_mismatch else ""
            lines.append(
                f"  first {'divergent' if op.status == DIVERGENT else 'suspect'} op: {op.op} (op {op.index} of {op.total} in {op.module}), own "
                f"error {_e(op.isolated.rel_err)} vs threshold {_e(op.threshold)}{why}"
            )
            ins = ", ".join(f"{d}{s}" for d, s in zip(op.input_dtypes, op.input_shapes))
            lines.append(f"    inputs: {ins or 'none'}"
                         + ("" if op.matched else "; this op does not run in the reference"))
        if op is not None and getattr(op, "kernel", None):
            k = op.kernel
            lines.append(f"    computed by {'library call' if k['extern'] else 'fused kernel'} "
                         f"{k['kernel']} ({', '.join(k['aten'])})")
        if r.stack is not None:
            s = r.stack
            for i in r.issues:
                if i.code in ("TS400", "TS401", "TS403"):
                    lines.append(f"  {i.code}: {i.message}")
            lines.append(f"  engine: compiled in {s.get('engine_seconds')} s, computes in "
                         f"{s.get('compute_dtype')}; localization: {s.get('localization')}"
                         + (f" ({s['localize_seconds']} s, {s['compiles']} compiles)"
                            if s.get("localize_seconds") else ""))
        if r.repro_path:
            lines.append(f"  repro: python {r.repro_path}")
        if r.op_repro_path:
            lines.append(f"  op repro: python {r.op_repro_path}")
        checked = [l for l in r.layers if l.replayed and l.isolated is not None]
        if r.stack is not None and len(checked) <= 1:
            lines.append("")
            continue
        bad = [l for l in r.layers if l.status != OK]
        if bad:
            lines.append(f"  modules over threshold: {len(bad)} of {len(checked)} checked")
            for l in bad[:top]:
                lines.append(
                    f"    {LABEL[l.status]:<9} {l.key:<40} own {_e(l.isolated.rel_err if l.isolated else None):>8}"
                    f"  accumulated {_e(l.propagated.rel_err if l.propagated else None):>8}"
                    f"  threshold {_e(l.threshold)}"
                )
        elif checked:
            worst = max(checked, key=lambda l: l.isolated.rel_err / l.threshold)
            lines.append(f"  all {len(checked)} checked modules within threshold"
                         f" (closest: {worst.key}, {_e(worst.isolated.rel_err if worst.isolated else None)}"
                         f" vs {_e(worst.threshold)})")
        lines.append("")
    overall = LABEL[report.status]
    lines.append(f"overall: {overall}")
    return "\n".join(lines)
