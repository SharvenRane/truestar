"""The README quickstart as a script, run against an installed truestar (not a checkout).

    python -I scripts/install_quickstart.py CHECKOUT_DIR

Used by .github/workflows/install-check.yml from a directory outside the checkout, after a plain
`pip install truestar-*.whl` into a fresh venv with CPU torch. It runs, in a scratch directory, what
README_DRAFT.md tells a new user to run ("Run it on your model" and "Install"), and checks the exit
codes the README promises (0 every backend matches, 2 usage error). Exits 0 when every step does
what the README says; otherwise prints the step and its output and exits 1.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

MODEL = textwrap.dedent('''
    import torch
    from torch import nn


    def get_model():      # an nn.Module
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 10))


    def get_inputs():     # a tensor, or a tuple of forward arguments
        torch.manual_seed(1)
        return torch.randn(8, 32)


    def get_kwargs():     # optional
        return {}
''')

failures = []


def step(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        failures.append(name)
        if detail:
            print(textwrap.indent(detail.rstrip(), "    "))


def run(args):
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


def main():
    checkout = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None

    import truestar

    where = os.path.abspath(truestar.__file__)
    print("truestar", truestar.__version__, "from", where)
    step("truestar is imported from the install, not the checkout",
         checkout is None or not where.startswith(checkout + os.sep), where)

    exe = shutil.which("truestar")
    step("the truestar command is on PATH", exe is not None, os.environ.get("PATH", ""))
    if exe is None:
        return
    print("truestar command:", exe)

    work = tempfile.mkdtemp(prefix="truestar-quickstart-")
    os.chdir(work)
    with open("my_model.py", "w", encoding="utf-8") as f:
        f.write(MODEL)

    # truestar check my_model.py --backends cpu
    rc, out, err = run([exe, "check", "my_model.py", "--backends", "cpu"])
    step("check my_model.py --backends cpu exits 0 and reports OK",
         rc == 0 and "overall: OK" in out, f"exit {rc}\n{out}\n{err}")

    # truestar check my_model.py --format agent: exactly one JSON document on stdout
    rc, out, err = run([exe, "check", "my_model.py", "--format", "agent"])
    try:
        doc = json.loads(out)
    except ValueError as e:
        doc = None
        step("check --format agent prints one JSON document", False, f"{e}\nexit {rc}\n{out}\n{err}")
    if doc is not None:
        cpu = [b for b in doc.get("backends", []) if b.get("name") == "cpu"]
        step("agent JSON has schema_version 1 and an exit_code equal to the process exit code",
             doc.get("schema_version") == 1 and doc.get("exit_code") == rc, out[:2000])
        step("agent JSON: the cpu backend matches the float64 reference",
             len(cpu) == 1 and cpu[0].get("status") == "ok", json.dumps(cpu, indent=1)[:2000])
        others = [(b["name"], b["status"]) for b in doc.get("backends", []) if b.get("name") != "cpu"]
        print("    other backends found on this machine:", others or "none")

    # truestar check my_model.py: every backend found on this machine, text output
    rc, out, err = run([exe, "check", "my_model.py"])
    expect = doc.get("exit_code") if doc else 0
    step("check my_model.py (every backend) gives the same exit code as the agent run",
         rc == expect and rc in (0, 1), f"exit {rc}, agent exit {expect}\n{out}\n{err}")

    # From Python: report = truestar.check(model, inputs); print(report.to_text())
    sys.path.insert(0, work)
    import my_model

    report = truestar.check(my_model.get_model(), my_model.get_inputs(), backends=["cpu"])
    text = report.to_text()
    print(textwrap.indent(text, "    | "))
    step("truestar.check(model, inputs) reports ok and renders text",
         report.status == "ok" and "overall: OK" in text, text)

    # Exit code 2 for a usage error
    rc, out, err = run([exe, "check", "no_such_model.py", "--backends", "cpu"])
    step("check on a missing model file exits 2", rc == 2, f"exit {rc}\n{out}\n{err}")

    # Install section: truestar check examples/cpu_check.py
    if checkout:
        example = os.path.join(checkout, "examples", "cpu_check.py")
        rc, out, err = run([exe, "check", example, "--backends", "cpu"])
        step("check examples/cpu_check.py exits 0", rc == 0, f"exit {rc}\n{out}\n{err}")


if __name__ == "__main__":
    main()
    if failures:
        print(f"\n{len(failures)} quickstart step(s) failed: {failures}")
        sys.exit(1)
    print("\nquickstart: every step did what README_DRAFT.md says")
