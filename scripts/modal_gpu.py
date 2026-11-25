"""Run Truestar tests (or a script, or a module) on a Modal cloud GPU and copy result files back.

usage (from any worktree):
  python scripts/modal_gpu.py <worktree> [--gpu L4] [--timeout 1800] -- <pytest args...>
  python scripts/modal_gpu.py <worktree> --script scripts/x.py -- <script args>
  python scripts/modal_gpu.py <worktree> --module truestar -- fingerprint --json out/fp.json
        [--fetch out] [--out-dir DIR] [--env K=V ...] [--commit SHA] [--max-usd 1.0] [--cpu 4]

Extends the shared helper: H100 is allowed, ``--module`` runs ``python -m``, ``--fetch PATH``
(relative to the remote working directory /w, a file or a directory) copies results back into
``--out-dir`` (default ./modal_out/<gpu>), the full stdout and stderr are saved there too, and
``--commit`` sets TRUESTAR_GIT_COMMIT for code that records its commit (the upload has no .git).

Cost: the estimate (GPU rate times the timeout, and times an expected time if ``--expect-s`` is
given) is printed before the run, times the GPU count for ``--gpu A10G:4``; ``--max-usd`` refuses a run whose timeout could cost more. Rates
are Modal's list prices per GPU second (modal.com/pricing, read 2026-11-24), GPU only.

Credentials come from ~/.secrets/modal_token_{id,secret}.txt and are never printed.
"""
import os
import pathlib
import sys
import time

HOME = pathlib.Path(os.path.expanduser("~"))
os.environ.setdefault("MODAL_TOKEN_ID", (HOME / ".secrets/modal_token_id.txt").read_text().strip())
os.environ.setdefault("MODAL_TOKEN_SECRET", (HOME / ".secrets/modal_token_secret.txt").read_text().strip())

import modal  # noqa: E402

USD_PER_S = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542, "H100": 0.001097}
ALLOWED_GPUS = set(USD_PER_S)


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(2)
    worktree = pathlib.Path(argv.pop(0)).resolve()
    gpu, timeout, script, module, out_dir, commit = "L4", 1800, None, None, None, None
    fetch, extra_env, max_usd, expect_s, cpu = [], {}, None, None, 4.0
    while argv and argv[0] != "--":
        flag = argv.pop(0)
        if flag == "--gpu":
            gpu = argv.pop(0)
        elif flag == "--timeout":
            timeout = int(argv.pop(0))
        elif flag == "--script":
            script = argv.pop(0)
        elif flag == "--module":
            module = argv.pop(0)
        elif flag == "--fetch":
            fetch.append(argv.pop(0))
        elif flag == "--out-dir":
            out_dir = pathlib.Path(argv.pop(0))
        elif flag == "--env":
            k, _, v = argv.pop(0).partition("=")
            extra_env[k] = v
        elif flag == "--commit":
            commit = argv.pop(0)
        elif flag == "--max-usd":
            max_usd = float(argv.pop(0))
        elif flag == "--expect-s":
            expect_s = float(argv.pop(0))
        elif flag == "--cpu":
            cpu = float(argv.pop(0))
        else:
            print("unknown flag", flag)
            sys.exit(2)
    rest = argv[1:] if argv and argv[0] == "--" else argv
    kind, _, count = gpu.partition(":")     # "A10G:4": four GPUs in one container
    count = int(count or 1)
    if kind not in ALLOWED_GPUS or not 1 <= count <= 4:
        print(f"gpu {gpu} not allowed; allowed: {sorted(ALLOWED_GPUS)}, with :2 or :4 for several")
        sys.exit(2)
    rate = USD_PER_S[kind] * count
    worst = rate * timeout
    line = f"[modal_gpu] {gpu} at ${rate * 3600:.2f}/h: at most ${worst:.2f} (timeout {timeout} s)"
    if expect_s:
        line += f", expected about ${rate * expect_s:.2f} ({expect_s:.0f} s)"
    print(line, flush=True)
    if max_usd is not None and worst > max_usd:
        print(f"[modal_gpu] refused: the timeout allows ${worst:.2f}, over --max-usd {max_usd}")
        sys.exit(2)
    out_dir = out_dir or pathlib.Path("modal_out") / gpu.replace(":", "x")
    if commit:
        extra_env.setdefault("TRUESTAR_GIT_COMMIT", commit)
        extra_env.setdefault("TRUESTAR_GIT_DIRTY", "false")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.11.0", "torchvision==0.26.0", "numpy==2.2.6", "pytest==8.3.3",
                     "numba==0.68.1", "onnx==1.17.0", "onnxscript", "onnxruntime")
        # the Windows fingerprints ran Triton 3.7.1; pin it so Triton is not the difference
        .pip_install("triton==3.7.1")
        .add_local_dir(str(worktree / "src"), "/w/src")
        .add_local_dir(str(worktree / "tests"), "/w/tests")
        .add_local_dir(str(worktree / "scripts"), "/w/scripts")
        .add_local_dir(str(worktree / "examples"), "/w/examples")
        .add_local_file(str(worktree / "pyproject.toml"), "/w/pyproject.toml")
    )
    app = modal.App("truestar-gpu-xchip")

    @app.function(image=image, gpu=gpu, timeout=timeout, serialized=True, memory=32768, cpu=cpu)
    def run(cmd: list, env_extra: dict, fetch_paths: list):
        import os as _os
        import pathlib as _pl
        import subprocess
        import sys as _sys
        import time as _time

        env = dict(_os.environ, PYTHONPATH="/w/src", **env_extra)
        cmd = [_sys.executable if c == "python" else c for c in cmd]
        _os.makedirs("/w/out", exist_ok=True)
        probe = subprocess.run([_sys.executable, "-c", "import torch, triton; print('torch', torch.__version__, "
                                "'triton', triton.__version__, 'cuda', torch.cuda.is_available(), "
                                "torch.cuda.get_device_name(0))"], capture_output=True, text=True, env=env)
        t0 = _time.time()
        p = subprocess.run(cmd, cwd="/w", env=env, capture_output=True, text=True)
        files = {}
        for fp in fetch_paths:
            path = _pl.Path("/w") / fp
            items = [q for q in path.rglob("*") if q.is_file()] if path.is_dir() else ([path] if path.exists() else [])
            for q in items:
                files[str(q.relative_to("/w"))] = q.read_bytes()
        return {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "probe": probe.stdout + probe.stderr,
                "seconds": _time.time() - t0, "files": files}

    if script:
        cmd = ["python", script] + rest
    elif module:
        cmd = ["python", "-m", module] + rest
    else:
        cmd = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"] + rest
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    t0 = time.time()
    with app.run():
        res = run.remote(cmd, extra_env, fetch)
    wall = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.txt").write_text(res["stdout"], encoding="utf-8")
    (out_dir / "stderr.txt").write_text(res["stderr"], encoding="utf-8")
    for rel, data in res["files"].items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(res["probe"].strip())
    print(res["stdout"][-6000:])
    if res["stderr"].strip():
        print("--- stderr tail ---")
        print(res["stderr"][-3000:])
    print(f"[modal_gpu] gpu={gpu} exit={res['code']} command {res['seconds']:.0f} s, wall {wall:.0f} s "
          f"(GPU cost of the wall time about ${rate * wall:.2f}); {len(res['files'])} file(s) in {out_dir}")
    sys.exit(res["code"])


if __name__ == "__main__":
    main()
