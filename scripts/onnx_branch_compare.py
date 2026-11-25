"""Compare torch.onnx.export and truestar export on models with Python branches (docs/EXPORT.md).

Run: PYTHONPATH=src python scripts/onnx_branch_compare.py   (needs onnx and onnxruntime, CPU only)

For each model and exporter it prints what the exporter did and, when it produced a file, whether
onnxruntime agrees with eager PyTorch on inputs that take the other side of the branch and on a
batch size different from the export batch.
"""

import io
import os
import re
import sys
import tempfile
import warnings

import torch
from torch import nn

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


class ValueBranch(nn.Module):
    """A Python if on a tensor value."""

    def forward(self, x):
        if x.mean() > 0:
            return torch.relu(x)
        return -x


class BatchBranch(nn.Module):
    """A Python if on the batch size."""

    def forward(self, x):
        if x.shape[0] > 4:
            return x * 2
        return x + 1


def _ort(path_or_bytes, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(path_or_bytes, providers=["CPUExecutionProvider"])
    return torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0])


def _short(e: BaseException, n: int = 3) -> str:
    text = re.sub(r"\[[0-9;]*m", "", str(e))
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return f"{type(e).__name__}: " + " | ".join(lines[:n])


def onnx_case(model, label, dynamo, dynamic, probes):
    x = torch.full((2, 3), 3.0)  # mean > 0, batch 2
    buf = os.path.join(tempfile.mkdtemp(), "m.onnx")
    print(f"\n[{label}] torch.onnx.export(dynamo={dynamo}, dynamic batch={dynamic}) at x=3*ones(2, 3)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            kw = {}
            if dynamic and not dynamo:
                kw = dict(input_names=["x"], dynamic_axes={"x": {0: "B"}})
            if dynamic and dynamo:
                kw = dict(input_names=["x"], dynamic_shapes=({0: torch.export.Dim("B", min=1, max=64)},))
            stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                torch.onnx.export(model, (x,), buf, dynamo=dynamo, **kw)
            finally:
                sys.stdout = stdout
            status = "wrote a file"
        except Exception as e:
            status = "raised " + _short(e)
        tracer = [str(m.message).splitlines()[0] for m in w if "Tracer" in type(m.message).__name__]
    print(f"  export: {status}")
    for t in tracer[:1]:
        print(f"  warning: {t[:160]}")
    if not os.path.exists(buf):
        return
    for name, inp in probes:
        want = model(inp)
        try:
            got = _ort(buf, inp)
            ok = got.shape == want.shape and torch.allclose(got, want)
            err = (got - want).abs().max().item() if got.shape == want.shape else float("nan")
            print(f"  onnxruntime {name}: {'matches eager' if ok else f'WRONG, max abs error {err:.3g}'}")
        except Exception as e:
            print(f"  onnxruntime {name}: raised {_short(e, 1)[:200]}")


def truestar_case(model, label):
    from truestar.export import ExportError, export
    print(f"\n[{label}] truestar export at x=3*ones(2, 3), dynamic batch")
    stdout = sys.stdout
    sys.stdout = io.StringIO()  # torch.export prints the partial graph on a data dependent error
    try:
        export(model, torch.full((2, 3), 3.0))
        msg = "exported"
    except ExportError as e:
        msg = str(e)
    finally:
        sys.stdout = stdout
    print("  " + msg.replace("\n", "\n  "))


def main():
    import onnx
    import onnxruntime
    print(f"torch {torch.__version__}, onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}")
    probes_value = [("x=-3*ones(2, 3), other side", torch.full((2, 3), -3.0)),
                    ("x=3*ones(7, 3), batch 7", torch.full((7, 3), 3.0))]
    probes_batch = [("x=3*ones(2, 3), same batch", torch.full((2, 3), 3.0)),
                    ("x=3*ones(7, 3), batch 7", torch.full((7, 3), 3.0))]
    for dynamo in (False, True):
        onnx_case(ValueBranch(), "ValueBranch", dynamo, False, probes_value)
        onnx_case(BatchBranch(), "BatchBranch", dynamo, True, probes_batch)
    truestar_case(ValueBranch(), "ValueBranch")
    truestar_case(BatchBranch(), "BatchBranch")


if __name__ == "__main__":
    main()
