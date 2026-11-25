"""Probe: does the CPU float16 backward of resnet18 depend on ATEN_CPU_CAPABILITY?

PyTorch and torchvision only, no Truestar. Run:

    python scripts/probe_cpu_scalar_fp16.py

Context: tests/test_gradcheck.py::test_clean_float16_resnet_reports_clean failed on some GitHub
ubuntu runners (TS180, layer4.1.bn2.weight). Locally it failed with ATEN_CPU_CAPABILITY=default and
passed with avx2 and avx512, which looked like a scalar float16 kernel bug.

The script runs one child process per capability (the variable must be set before torch loads) and
prints, for each:

* draw: a hash of the test's own data (torch.manual_seed(0), resnet18 init, torch.randn input).
  normal_ takes another code path per capability, so the bits of the draw differ.
* own draw: the error of the float16 gradient of layer4.1.bn2.weight against float64, on the
  capability's own draw (what the test saw), and the ReLU mask flips at layer4.1.
* fixed draw: the same numbers on ONE draw shared by every child (the DEFAULT draw, saved to a temp
  file), plus a hash of all float16 gradients, so the kernels themselves are compared bit for bit,
  plus the error of a correctly rounded float16 run (float64 math, every op result rounded to
  float16) against float64. If the float16 kernels match the rounded run, they are right and the
  error is float16's own on that input.
"""
import hashlib
import os
import platform
import subprocess
import sys
import tempfile

CAPS = ["default", "avx2", "avx512"]


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def child(fixed_path: str) -> None:
    import contextlib

    import torch
    import torchvision
    from torch import nn
    from torch.utils._python_dispatch import TorchDispatchMode

    def h(*ts) -> str:
        m = hashlib.sha1()
        for t in ts:
            m.update(t.detach().contiguous().cpu().numpy().tobytes())
        return m.hexdigest()[:10]

    class Rounded(TorchDispatchMode):
        """float64 math with every floating op result rounded to float16 (in place results too)."""

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            schema = func._schema
            rnd = lambda t: t.half().double() if isinstance(t, torch.Tensor) and t.dtype == torch.float64 else t
            if any(r.alias_info is not None for r in schema.returns):
                if schema.is_mutable:
                    for t in (out if isinstance(out, (tuple, list)) else [out]):
                        if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
                            t.copy_(t.half().double())
                return out
            if isinstance(out, (tuple, list)):
                return type(out)(rnd(t) for t in out)
            return rnd(out)

    def own_draw():
        torch.manual_seed(0)
        m = torchvision.models.resnet18(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        return m.state_dict(), x

    target = torch.randn(2, 10, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    def run(sd, x, kind):
        m = torchvision.models.resnet18(num_classes=10).eval()
        m.load_state_dict(sd)
        pre = {}
        m.layer4[1].relu.inplace = False
        m.layer4[1].relu.register_forward_hook(
            lambda mod, i, o: pre.setdefault("in", []).append(i[0].detach().double()))
        if kind == "half":
            m, xi, ctx = m.half(), x.half(), contextlib.nullcontext()
        else:
            m, xi = m.double(), x.double()
            ctx = contextlib.nullcontext()
            if kind == "rounded":
                with torch.no_grad():
                    for p in list(m.parameters()) + list(m.buffers()):
                        if p.is_floating_point():
                            p.copy_(p.half().double())
                xi, ctx = xi.half().double(), Rounded()
        with ctx:
            y = m(xi)
            ((y.double() - target) ** 2).mean().backward()
        grads = {n: p.grad.detach().double().clone() for n, p in m.named_parameters()}
        return grads, pre["in"][-1]  # second ReLU call of the block: after bn2 + identity

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    def report(tag, sd, x):
        g64, p64 = run(sd, x, "double")
        g16, p16 = run(sd, x, "half")
        gr, _ = run(sd, x, "rounded")
        k = "layer4.1.bn2.weight"
        flips = int(((p64 > 0) != (p16 > 0)).sum())
        print(f"  {tag:10s} bn2.weight grad: float16 vs float64 {rel(g16[k], g64[k]):.3e}, "
              f"rounded run vs float64 {rel(gr[k], g64[k]):.3e}, float16 vs rounded run "
              f"{rel(g16[k], gr[k]):.3e}; layer4.1 ReLU mask flips {flips}; float16 grads hash "
              f"{h(*[g16[n] for n in sorted(g16)])}")

    sd, x = own_draw()
    print(f"capability {torch.backends.cpu.get_cpu_capability()} "
          f"(ATEN_CPU_CAPABILITY={os.environ.get('ATEN_CPU_CAPABILITY', '<unset>')})")
    print(f"  draw hash  input {h(x)}, conv1.weight {h(sd['conv1.weight'])}, "
          f"layer4.1.bn2.weight {h(sd['layer4.1.bn2.weight'])}")
    report("own draw", sd, x)
    if not os.path.exists(fixed_path):
        torch.save({"sd": sd, "x": x}, fixed_path)
    d = torch.load(fixed_path)
    report("fixed draw", d["sd"], d["x"])


def main() -> None:
    import torch
    print(f"torch {torch.__version__}, python {platform.python_version()}, {platform.system()} "
          f"{platform.machine()}, CPU {cpu_model()}")
    print(f"this process (no override): capability {torch.backends.cpu.get_cpu_capability()}")
    with tempfile.TemporaryDirectory() as tmp:
        fixed = os.path.join(tmp, "fixed_draw.pt")  # written by the first child (DEFAULT)
        for cap in CAPS:
            env = dict(os.environ, ATEN_CPU_CAPABILITY=cap)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", fixed],
                               env=env, capture_output=True, text=True)
            print(r.stdout.rstrip())
            if r.returncode:
                print(f"  child failed ({r.returncode}): {r.stderr.strip()[-500:]}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
