"""Run scripts/bench_heldout.py on a Modal L4 (extends the shared modal_gpu helper: a CUDA devel image
so torch.utils.cpp_extension can build the kernels, the held out data uploaded, results and the
extension cache kept in a Modal Volume across runs, and each run bounded by --deadline).

    python -I scripts/modal_heldout.py --probe
    python -I scripts/modal_heldout.py [--only REGEX] [--workers 4] [--deadline 480]
                                       [--out out] [--items results/heldout_runs/items.json]

Results are copied to results/heldout_runs/<out>/<id>.json (Modal Volume folder /vol/<out>).
"""
import json
import os
import pathlib
import sys

HOME = pathlib.Path(os.path.expanduser("~"))
os.environ.setdefault("MODAL_TOKEN_ID", (HOME / ".secrets/modal_token_id.txt").read_text().strip())
os.environ.setdefault("MODAL_TOKEN_SECRET",
                      (HOME / ".secrets/modal_token_secret.txt").read_text().strip())

import modal  # noqa: E402

WT = pathlib.Path(__file__).resolve().parent.parent


def main():
    args = sys.argv[1:]
    probe = "--probe" in args
    opts = {"--only": None, "--workers": "4", "--deadline": "480", "--out": "out",
            "--items": str(WT / "results" / "heldout_runs" / "items.json")}
    for k in list(opts):
        if k in args:
            opts[k] = args[args.index(k) + 1]
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .pip_install("torch==2.11.0", "numpy==2.2.6", "ninja",
                     index_url="https://download.pytorch.org/whl/cu128",
                     extra_index_url="https://pypi.org/simple")
        .add_local_dir(str(WT / "src"), "/w/src")
        .add_local_dir(str(WT / "scripts"), "/w/scripts")
        .add_local_dir(str(WT / "tests" / "data" / "heldout"), "/w/tests/data/heldout")
        .add_local_dir(str(WT / "results" / "heldout_metr"), "/w/results/heldout_metr")
        .add_local_file(opts["--items"], "/w/items.json")
    )
    vol = modal.Volume.from_name("truestar-heldout", create_if_missing=True)
    app = modal.App("truestar-heldout")

    @app.function(image=image, gpu="L4", cpu=4, memory=24576, timeout=600, serialized=True,
                  volumes={"/vol": vol})
    def run(probe: bool, only, workers: str, deadline: str, outname: str = "out"):
        import glob as _g
        import subprocess
        import os as _os
        import sys as _sys
        env = dict(_os.environ, PYTHONPATH="/w/src", HELDOUT_EXT_DIR="/vol/ext",
                   TORCH_CUDA_ARCH_LIST="8.9")
        if probe:
            cmds = [["nvcc", "--version"],
                    [_sys.executable, "-c", "import torch,triton;print(torch.__version__, "
                     "torch.version.cuda, triton.__version__, torch.cuda.get_device_name(0))"]]
            outs = [subprocess.run(c, capture_output=True, text=True, env=env) for c in cmds]
            return {"probe": "\n".join(o.stdout[-800:] + o.stderr[-800:] for o in outs)}, ""
        _os.makedirs(f"/vol/{outname}", exist_ok=True)
        cmd = [_sys.executable, "/w/scripts/bench_heldout.py", "--run", "/w/items.json",
               "--out-dir", f"/vol/{outname}", "--workers", workers, "--deadline", deadline]
        if only:
            cmd += ["--only", only]
        p = subprocess.run(cmd, cwd="/w", env=env, capture_output=True, text=True)
        vol.commit()
        res = {}
        for f in _g.glob(f"/vol/{outname}/*.json"):
            res[_os.path.basename(f)] = open(f, encoding="utf-8").read()
        return res, p.stdout[-6000:] + "\n" + p.stderr[-3000:]

    @app.function(image=image, cpu=32, memory=65536, timeout=600, serialized=True,
                  volumes={"/vol": vol})
    def compile_all(only, workers: str, deadline: str):
        """CPU container: build every candidate's extension into the shared cache only."""
        import subprocess
        import os as _os
        import sys as _sys
        env = dict(_os.environ, PYTHONPATH="/w/src", HELDOUT_EXT_DIR="/vol/ext",
                   TORCH_CUDA_ARCH_LIST="8.9")
        cmd = [_sys.executable, "/w/scripts/bench_heldout.py", "--run", "/w/items.json",
               "--out-dir", "/vol/compiled", "--workers", workers, "--deadline", deadline,
               "--compile-only", "--timeout", "420"]
        if only:
            cmd += ["--only", only]
        p = subprocess.run(cmd, cwd="/w", env=env, capture_output=True, text=True)
        vol.commit()
        return p.stdout[-4000:] + "\n" + p.stderr[-2000:]

    sys.stdout.reconfigure(encoding="utf-8")
    if "--compile" in args:
        with app.run():
            print(compile_all.remote(opts["--only"], opts["--workers"], opts["--deadline"]))
        return 0
    with app.run():
        res, log = run.remote(probe, opts["--only"], opts["--workers"], opts["--deadline"],
                              opts["--out"])
    if probe:
        print(res["probe"])
        return 0
    OUT = WT / "results" / "heldout_runs" / opts["--out"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in res.items():
        (OUT / name).write_text(text, encoding="utf-8")
    print(log)
    print(f"[modal_heldout] {len(res)} result files in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
