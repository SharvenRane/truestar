"""Run the serving battery (src/truestar/conform/serving_battery.py) on a Modal L4 and bring the results home.

usage:
  <python with modal> scripts/modal_serving.py vllm   [--dtype bfloat16] [--out results/serving] [--settings base,...] [--plant KEY=JSON]
  <python with modal> scripts/modal_serving.py sglang [--dtype bfloat16] [--out results/serving]
then, on any machine:
  truestar conform vllm results/serving/vllm_bfloat16

Modal credentials: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET, or ~/.secrets/modal_token_{id,secret}.txt.
Each engine gets its own image (they pin different torch builds). The model is built in the container
from the spec in code; HF_HUB_OFFLINE=1 makes any download attempt fail instead of happening.
Only the worktree's src/ is uploaded. The weights are not brought back (the judge rebuilds them from
the seed and checks their hash).
"""
import io
import os
import pathlib
import sys
import tarfile

HOME = pathlib.Path(os.path.expanduser("~"))
for var, name in (("MODAL_TOKEN_ID", "modal_token_id.txt"), ("MODAL_TOKEN_SECRET", "modal_token_secret.txt")):
    if var not in os.environ and (HOME / ".secrets" / name).is_file():
        os.environ[var] = (HOME / ".secrets" / name).read_text().strip()

import modal  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PINS = {  # engine -> pip requirements (each engine brings its own torch)
    "vllm": ["vllm==0.34.0", "safetensors"],
    "sglang": ["sglang==0.5.23", "safetensors"],
}
APT = {"vllm": [], "sglang": ["libnuma1", "libnuma-dev"]}
BASE = {"vllm": None, "sglang": "nvidia/cuda:13.0.2-devel-ubuntu24.04"}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in PINS:
        print(__doc__)
        sys.exit(2)
    engine = argv.pop(0)
    dtype, out, settings, timeout, plant = "bfloat16", str(ROOT / "results" / "serving"), None, 1800, None
    while argv:
        flag = argv.pop(0)
        if flag == "--dtype":
            dtype = argv.pop(0)
        elif flag == "--out":
            out = argv.pop(0)
        elif flag == "--settings":
            settings = argv.pop(0)
        elif flag == "--plant":
            plant = argv.pop(0)
        elif flag == "--timeout":
            timeout = int(argv.pop(0))
        else:
            print("unknown flag", flag)
            sys.exit(2)
    if BASE[engine]:  # FlashInfer compiles kernels when the engine starts, so nvcc must be in the image
        image = modal.Image.from_registry(BASE[engine], add_python="3.12")
    else:
        image = modal.Image.debian_slim(python_version="3.12")
    if APT[engine]:
        image = image.apt_install(*APT[engine])
    image = (image.uv_pip_install(*PINS[engine])
             .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING"})
             .add_local_dir(str(ROOT / "src"), "/w/src"))
    app = modal.App(f"truestar-serving-{engine}")

    @app.function(image=image, gpu="L4", timeout=timeout, serialized=True)
    def run(engine: str, dtype: str, settings, plant):
        import subprocess, sys as _sys, os as _os, io as _io, tarfile as _tar
        env = dict(_os.environ, PYTHONPATH="/w/src")
        cmd = [_sys.executable, "-m", "truestar.conform.serving_battery", "run", "--engine", engine,
               "--out", "/tmp/res", "--dtype", dtype] + (["--settings", settings] if settings else [])
        cmd += ["--plant", plant] if plant else []
        p = subprocess.run(cmd, cwd="/w", env=env, capture_output=True, text=True)
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w:gz") as t:
            if _os.path.isdir("/tmp/res"):
                t.add("/tmp/res", arcname="res", filter=lambda ti: None if ti.name.endswith(".safetensors") else ti)
        return p.returncode, p.stdout[-5000:], p.stderr[-8000:], buf.getvalue()

    sys.stdout.reconfigure(encoding="utf-8")
    with app.run():
        code, so, se, blob = run.remote(engine, dtype, settings, plant)
    dest = pathlib.Path(out) / (f"{engine}_{dtype}" + ("_planted" if plant else ""))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.isfile() or ".." in m.name or m.name.startswith("/"):
                continue
            rel = pathlib.Path(m.name).relative_to("res")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(t.extractfile(m).read())
    print(so)
    if se.strip():
        print("--- stderr tail ---")
        print(se[-3000:])
    print(f"[modal_serving] engine={engine} dtype={dtype} exit={code} results={dest}")
    sys.exit(code)


if __name__ == "__main__":
    main()
