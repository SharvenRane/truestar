"""The compile report: what became a Truestar kernel and what did not, with the reason.

Nothing in a compiled program runs on a path that is not listed here. ``report.fallbacks`` holds
every op that ran through the float64 reference implementation, ``report.library`` every op that
ran through a torch library kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

KINDS = ("kernel", "library", "fallback", "host", "layout", "create", "adapter", "cond", "loop", "random", "recompute", "offload", "fence",
         "reload", "wait")


def _walk(entries: List[Dict[str, Any]], path: str = ""):
    for e in entries:
        yield path, e
        for label, sub in e.get("regions", {}).items():
            yield from _walk(sub, f"{path}{e['name']}/{label}/")


@dataclass
class CompileReport:
    program: str
    device: str
    dtype: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    compile_seconds: float = 0.0
    kernel_cache: Dict[str, int] = field(default_factory=dict)
    # float32 dot products and convolutions: "ieee" or "tf32", plus how that was decided
    fp32_precision: str = "ieee"
    fp32_precision_source: str = "pinned"
    # mixed precision islands of a 16 bit program: (value, storage dtype, shape), kept in float32
    islands: List[Any] = field(default_factory=list)

    @classmethod
    def from_program(cls, cp) -> "CompileReport":
        return cls(cp.program.name, cp.opts.device, cp.opts.dtype, cp.plan.report_entries(),
                   cp.compile_seconds, dict(cp.kernel_cache),
                   "tf32" if cp.opts.allow_tf32 else "ieee",
                   "follows torch" if getattr(cp, "follows_tf32", False) else "pinned",
                   list(getattr(cp.plan, "islands", [])))

    def of_kind(self, kind: str) -> List[Dict[str, Any]]:
        out = []
        for path, e in _walk(self.entries):
            if e["kind"] == kind:
                out.append(dict(e, region=path.rstrip("/")))
        return out

    @property
    def kernels(self) -> List[Dict[str, Any]]:
        return self.of_kind("kernel")

    @property
    def fallbacks(self) -> List[Dict[str, Any]]:
        return self.of_kind("fallback")

    @property
    def library(self) -> List[Dict[str, Any]]:
        return self.of_kind("library")

    @property
    def host(self) -> List[Dict[str, Any]]:
        return self.of_kind("host")

    def kernel_ops(self) -> List[str]:
        """IR ops whose arithmetic runs in Truestar kernels (with repeats)."""
        return [op for k in self.kernels for op in k["ops"]]

    def counts(self) -> Dict[str, int]:
        c = {k: 0 for k in KINDS}
        for _, e in _walk(self.entries):
            c[e["kind"]] += 1
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {"schema": "truestar.compiler.report/1", "program": self.program, "device": self.device,
                "dtype": self.dtype, "compile_seconds": self.compile_seconds, "kernel_cache": self.kernel_cache,
                "fp32_precision": self.fp32_precision, "fp32_precision_source": self.fp32_precision_source,
                "islands": [{"value": n, "dtype": d, "shape": list(sh)} for n, d, sh in self.islands],
                "counts": self.counts(),
                "fallbacks": [{k: f[k] for k in ("name", "ops", "reason", "source", "path", "region")}
                              for f in self.fallbacks],
                "library": [{k: f.get(k) for k in ("name", "ops", "impl", "region", "reason")}
                            for f in self.library],
                "entries": self.entries}

    def to_text(self) -> str:
        c = self.counts()
        shapes: Dict[str, int] = {}
        for k in self.kernels:
            shapes[k["shape"]] = shapes.get(k["shape"], 0) + 1
        lines = [f"compile report: {self.program} on {self.device}, {self.dtype} "
                 f"(compiled in {self.compile_seconds:.2f} s)"
                 + (f"; float32 dot products {self.fp32_precision.upper() if self.fp32_precision == 'tf32' else 'IEEE'}"
                    f" ({self.fp32_precision_source})" if self.dtype == "float32" and self.device == "cuda" else ""),
                 f"  Truestar kernels: {c['kernel']} covering {len(self.kernel_ops())} IR ops"
                 + (" (" + ", ".join(f"{n} {s}" for s, n in sorted(shapes.items())) + ")" if shapes else ""),
                 f"  library kernels (torch): {c['library']}",
                 f"  random draws (keyed generator kernel, docs/RANDOM.md): {c['random']}",
                 f"  fallbacks (float64 reference implementation on the CPU): {c['fallback']}",
                 f"  host scalars (integer bookkeeping on the CPU): {c['host']}",
                 f"  layout ops (views and copies, no arithmetic): {c['layout']}; constants created: "
                 f"{c['create']}; dtype adapters: {c['adapter']}; cond: {c['cond']}; loop: {c['loop']}"]
        for f in self.fallbacks:
            where = f" in {f['region']}" if f["region"] else ""
            lines.append(f"  FALLBACK {f['ops'][0]} {f['name']}{where}: {f['reason']}"
                         + (f" ({f['source']})" if f.get("source") else ""))
        for f in self.library:
            where = f" in {f['region']}" if f["region"] else ""
            why = f" ({f['reason']})" if f.get("reason") else ""
            lines.append(f"  LIBRARY {f['ops'][0]} {f['name']}{where}: {f['impl']}{why}")
        lines.append("  steps:")

        def show(entries, indent):
            for e in entries:
                extra = ""
                if e["kind"] == "kernel":
                    extra = f" [{e['shape']}] " + " + ".join(e["ops"])
                elif e["kind"] in ("library", "fallback", "host", "layout", "create"):
                    extra = " " + " ".join(e["ops"])
                elif e["kind"] == "random":
                    extra = f" {e.get('site')} via {e.get('impl')}"
                lines.append(f"{indent}{e['kind']:<8} {e['name']}{extra}")
                for label, sub in e.get("regions", {}).items():
                    lines.append(f"{indent}  {label}:")
                    show(sub, indent + "    ")

        show(self.entries, "    ")
        if self.dtype in ("float16", "bfloat16"):
            lines.insert(3, f"  mixed precision islands (float32 in a {self.dtype} program): {len(self.islands)}"
                         + (" (" + ", ".join(f"{n} {list(sh)}" for n, _, sh in self.islands[:8])
                            + (", ..." if len(self.islands) > 8 else "") + ")" if self.islands else ""))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()
