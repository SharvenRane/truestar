"""Run Truestar tests (or a script) on a Modal cloud GPU instead of the shared desktop GPU.

Copy of <research>/tools/modal_gpu.py with --pip PKG (repeatable) and --cmd CMD
(repeatable, a shell command run in the image after the installs), for benchmarks that need a
package Truestar does not depend on (scripts/bench_decode_flashinfer.py: flashinfer-python).

usage (from any worktree):
  python scripts/modal_gpu_extra.py <worktree> [--gpu L4] [--timeout 1800] -- <pytest args...>
  python scripts/modal_gpu_extra.py <worktree> --script scripts/x.py -- <script args>

Credentials come from ~/.secrets/modal_token_{id,secret}.txt. Spend is capped by the workspace usage limit
($30/month, credits included). Keep runs short: L4 is about $0.80/hour; avoid A100/H100 for these runs.
The worktree's src/ and tests/ and scripts/ are uploaded (no .git, no results, no large files).
Output: pytest/script stdout and exit code, printed locally.
"""
import os, sys, pathlib

HOME = pathlib.Path(os.path.expanduser("~"))
os.environ.setdefault("MODAL_TOKEN_ID", (HOME / ".secrets/modal_token_id.txt").read_text().strip())
os.environ.setdefault("MODAL_TOKEN_SECRET", (HOME / ".secrets/modal_token_secret.txt").read_text().strip())

import modal  # noqa: E402

ALLOWED_GPUS = {"T4", "L4", "A10G", "L40S"}


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__); sys.exit(2)
    worktree = pathlib.Path(argv.pop(0)).resolve()
    gpu, timeout, script, pips, cmds = "L4", 1800, None, [], []
    while argv and argv[0] != "--":
        flag = argv.pop(0)
        if flag == "--gpu":
            gpu = argv.pop(0)
        elif flag == "--timeout":
            timeout = int(argv.pop(0))
        elif flag == "--pip":
            pips.append(argv.pop(0))
        elif flag == "--cmd":
            cmds.append(argv.pop(0))
        elif flag == "--script":
            script = argv.pop(0)
        else:
            print("unknown flag", flag); sys.exit(2)
    rest = argv[1:] if argv and argv[0] == "--" else argv
    if gpu not in ALLOWED_GPUS:
        print(f"gpu {gpu} not allowed without the owner's yes; allowed: {sorted(ALLOWED_GPUS)}"); sys.exit(2)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        # torch brings its own matching Linux triton, so triton is not pinned here
        .pip_install("torch==2.11.0", "numpy==2.2.6", "pytest==8.3.3", "numba==0.68.1",
                     "onnx==1.17.0", "onnxscript", "onnxruntime")
    )
    if pips:
        image = image.pip_install(*pips)
    if cmds:
        image = image.run_commands(*cmds)
    image = (
        image
        .add_local_dir(str(worktree / "src"), "/w/src")
        .add_local_dir(str(worktree / "tests"), "/w/tests")
        .add_local_dir(str(worktree / "scripts"), "/w/scripts")
        .add_local_file(str(worktree / "pyproject.toml"), "/w/pyproject.toml")
    )
    app = modal.App("truestar-gpu-tests")

    @app.function(image=image, gpu=gpu, timeout=timeout, serialized=True)
    def run(cmd: list):
        import subprocess, os as _os, sys as _sys
        env = dict(_os.environ, PYTHONPATH="/w/src")
        cmd = [_sys.executable if c == "python" else c for c in cmd]
        probe = subprocess.run([_sys.executable, "-c", "import pytest, torch; print('pytest', pytest.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
                               capture_output=True, text=True, env=env)
        if probe.returncode != 0:
            return probe.returncode, probe.stdout, "image check failed: " + probe.stderr[-3000:]
        p = subprocess.run(cmd, cwd="/w", env=env, capture_output=True, text=True)
        return p.returncode, p.stdout[-20000:], p.stderr[-5000:]

    cmd = (["python", script] + rest) if script else (["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"] + rest)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    with app.run():
        code, out, err = run.remote(cmd)
    print(out)
    if err.strip():
        print("--- stderr tail ---"); print(err[-3000:])
    print(f"[modal_gpu] gpu={gpu} exit={code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
