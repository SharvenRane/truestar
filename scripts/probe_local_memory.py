"""Does a contraction config that spills to local memory fail when the GPU is out of memory?

    PYTHONPATH=src python scripts/probe_local_memory.py

Found 2026-11-24 on an RTX 5060 Laptop GPU (8 GB): a sweep that held about 40 CUDA graphs at once
crashed the CUDA context ("unspecified launch failure", driver event 13 "SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE
failed"). Two strict float32 tiles spill registers to local memory there (128 x 256 x 16 with 8
warps, 128 x 128 x 16 with 4 warps). Each case runs in its own child process, so a crash is
recorded instead of ending the probe: the config, whether the GPU was first filled to leave only
``--free-mb`` MiB, and whether the kernel was launched directly or replayed from a CUDA graph.
"""

import argparse
import json
import subprocess
import sys

CASES = [((128, 256, 16, 8, 3), "spills"), ((128, 128, 16, 4, 3), "spills"), ((64, 128, 16, 4, 3), "no spill")]


def child(cfg, fill, graph, free_mb):
    import torch
    from truestar import kernel as K
    from truestar.kernel import Graph, triton_backend as tb
    M, N, Kd = 2048, 4096, 1024
    g = Graph()
    x = g.input("x", (K.sym("M"), Kd), "float32")
    w = g.input("w", (N, Kd), "float32")
    b = g.input("b", (N,), "float32")
    (_, kern), = g.fuse({"y": g.relu(x @ g.transpose(w, 0, 1) + b)})
    X, W, B = (torch.randn(M, Kd, device="cuda"), torch.randn(N, Kd, device="cuda"),
               torch.randn(N, device="cuda"))
    ck = K.compile(kern, "cuda", tune=False, _config=tb._cfg(*cfg))
    hog = []
    if fill:                                     # leave about free_mb MiB of the GPU free
        while True:
            free = torch.cuda.mem_get_info()[0]
            if free <= free_mb * 2 ** 20 + 64 * 2 ** 20:
                break
            hog.append(torch.empty(min(free - free_mb * 2 ** 20, 256 * 2 ** 20), dtype=torch.uint8, device="cuda"))
    run = lambda: ck(x=X, w=W, b=B)              # noqa: E731
    if graph:
        run()
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            run()
        for _ in range(20):
            cg.replay()
    else:
        for _ in range(20):
            run()
    torch.cuda.synchronize()
    return {"free_mb_at_launch": torch.cuda.mem_get_info()[0] / 2 ** 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--free-mb", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.child:
        c = json.loads(a.child)
        print("RESULT " + json.dumps(child(tuple(c["cfg"]), c["fill"], c["graph"], a.free_mb)))
        return
    rows = []
    for cfg, kind in CASES:
        for fill in (False, True):
            for graph in (False, True):
                arg = json.dumps({"cfg": cfg, "fill": fill, "graph": graph})
                r = subprocess.run([sys.executable, __file__, "--child", arg, "--free-mb", str(a.free_mb)],
                                   capture_output=True, text=True, timeout=600)
                line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
                err = "" if line else (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
                row = {"config": "%dx%dx%d w%d s%d" % cfg, "kind": kind, "gpu_filled": fill,
                       "cuda_graph": graph, "ok": line is not None, "exit": r.returncode, "error": err}
                if line:
                    row.update(json.loads(line[7:]))
                rows.append(row)
                print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
