"""Single op repro for an upgrade regression: the op, its recorded inputs and recorded output."""

from __future__ import annotations

import os

import torch

from ..repro import _slug

OP_SCRIPT = '''"""truestar diff op repro: {op} from module {module} in {model}.
Reference: {source}. Inputs: {inputs}.
Recorded with torch {torch_rec} on {device_rec}; flagged with torch {torch_cur}.
Run: python op_repro.py [device]. Exit code 1 = the op's result moved beyond {band:.1e}."""
import os, sys, torch
p = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_payload.pt"), weights_only=False)
dev = sys.argv[1] if len(sys.argv) > 1 else {device!r}
def mv(o):
    if isinstance(o, torch.Tensor): return o.to(dev)
    if isinstance(o, torch.device): return torch.device(dev)
    if isinstance(o, (list, tuple)): return type(o)(mv(x) for x in o)
    return {{k: mv(v) for k, v in o.items()}} if isinstance(o, dict) else o
with torch.no_grad():
    out = torch.ops.{op}(*mv(p["args"]), **mv(p["kwargs"]))
outs = [out] if isinstance(out, torch.Tensor) else [t for t in out if isinstance(t, torch.Tensor)]
def rel(a, b):
    a, b = a.detach().cpu().double(), b.double()
    if not torch.equal(a.isfinite(), b.isfinite()): return float("inf")
    m = b.isfinite(); d, n = (a - b)[m].norm().item(), b[m].norm().item()
    return 0.0 if d == 0 else (d / n if n else float("inf"))
err = max([rel(a, b) for a, b in zip(outs, p["reference"])] or [0.0])
print(f"torch {{torch.__version__}} {op} on {{dev}}: relative change {{err:.3e}} (band {band:.1e})")
sys.exit(1 if err > {band!r} else 0)
'''


def write_op_repro(repro_dir: str, rep, device: str) -> str:
    op = rep.blamed_op
    out_dir = os.path.join(repro_dir, _slug(f"diff_{op.module}"))
    os.makedirs(out_dir, exist_ok=True)
    payload = {"op": op.op, "args": op.args, "kwargs": op.kwargs,
               "reference": [t.double() if t.is_floating_point() else t for t in op.reference]}
    torch.save(payload, os.path.join(out_dir, "op_payload.pt"))
    inputs = ", ".join(f"{d}{s}" for d, s in zip(op.input_dtypes, op.input_shapes)) or "none"
    source = ("the output this op gave in the recorded environment" if op.source ==
              "recorded_trace" else "float64 CPU math")
    script = OP_SCRIPT.format(op=op.op, module=op.module, model=rep.model, source=source,
                              inputs=inputs, torch_rec=rep.env_recorded.get("torch"),
                              device_rec=rep.env_recorded.get("gpu.name") or "cpu",
                              torch_cur=rep.env_current.get("torch"), band=op.band,
                              device=device)
    path = os.path.join(out_dir, "op_repro.py")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    return path
