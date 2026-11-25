"""Which launch config parameters keep a kernel's summation order (bitwise equal results)?

    PYTHONPATH=src python scripts/order_classes.py --json out.json

For each loop shape whose bits can follow its config (attention, tile_single, tile_looped), the script
builds kernels of several kinds and sizes, runs every config of a grid (block sizes, warps, stages and
the attention switches, each one ``check_config`` accepts), and groups the configs by bitwise equal
outputs. A parameter that varies inside a group does not change the bits for that case; one that
splits groups does. The order key in triton_backend (``order_key``) is checked against every group:
configs with the same key must give the same bits.
"""

import argparse
import itertools
import json


def cases():
    from truestar import kernel as K
    from truestar.kernel import Graph
    out = []
    for dtype in ("float16", "float32"):
        for causal, T, N, D in ((True, 300, 300, 64), (False, 1, 2500, 128), (False, 200, 700, 64)):
            g = Graph()
            Ty, Ny = K.sym("T"), K.sym("N")
            q = g.input("q", (2, 4, Ty, D), dtype)
            k, v = (g.input(n, (2, 4, Ny, D), dtype) for n in "kv")
            s = (q @ k.T) * (D ** -0.5)
            if causal:
                s = g.pointwise(lambda s, b, h, i, j: K.where(j <= i + (N - T), s, K.NEG_INF), s, with_index=True)
            (_, kern), = g.fuse({"o": g.softmax(s) @ v})
            out.append((f"attention {dtype} {'causal' if causal else 'full'} T{T} N{N} D{D}", kern,
                        {"q": (2, 4, T, D), "k": (2, 4, N, D), "v": (2, 4, N, D)}, dtype))
        for rows, cols in ((64, 1000), (300, 4096), (8, 20000)):
            g = Graph()
            R = K.sym("R")
            x = g.input("x", (R, cols), dtype)
            (_, kern), = g.fuse({"y": g.softmax(x)})
            out.append((f"softmax {dtype} {rows}x{cols}", kern, {"x": (rows, cols)}, dtype))
            g = Graph()
            x = g.input("x", (R, cols), dtype)
            (_, kern), = g.fuse({"y": g.layer_norm(x)})
            out.append((f"layer norm {dtype} {rows}x{cols}", kern, {"x": (rows, cols)}, dtype))
    return out


def grid_for(tb, p, env):
    kind = tb.config_kind(p, env)
    d = tb.default_config(p, env)
    if kind == "attention":
        vals = dict(BM=(16, 32, 64, 128), BN=(32, 64, 128), num_warps=(2, 4, 8), num_stages=(2, 3),
                    LPT=(True, False), SPLIT=(True, False), KV_CHUNK=(0, 256, 512))
    elif kind == "tile_single":
        vals = dict(BR=(1, 2, 4, 8, 16), num_warps=(2, 4, 8, 16))
    elif kind == "tile_looped":
        vals = dict(BC=(512, 1024, 2048, 4096), num_warps=(2, 4, 8))
    else:
        return [d]
    keys = list(vals)
    out = [d]
    for combo in itertools.product(*(vals[k] for k in keys)):
        c = dict(d, **dict(zip(keys, combo)))
        if c not in out and tb.check_config(p, env, c) is None:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--limit", type=int, default=400, help="configs per case at most")
    a = ap.parse_args()
    import torch
    from truestar import kernel as K
    from truestar.kernel import ir
    from truestar.kernel import triton_backend as tb

    report = []
    for name, kern, shapes, dtype in cases():
        gen = torch.Generator(device="cuda").manual_seed(0)
        dt = getattr(torch, dtype)
        ins = {n: torch.randn(*s, device="cuda", generator=gen).to(dt) for n, s in shapes.items()}
        base = K.compile(kern, "cuda", tune=False, deterministic=True)
        ref = base(**ins)
        p = base.art.plan
        env = dict(base.launch_configs()[0][0]) if False else None
        sig_env = {}
        ck_env = base._fast[next(iter(base._fast))].env
        sig_env.update(ck_env)
        configs = grid_for(tb, p, sig_env)[: a.limit]
        groups = []          # [bits tensor, [configs]]
        for c in configs:
            try:
                out = K.compile(kern, "cuda", tune=False, deterministic=True, _config=c)(**ins)
            except Exception as e:   # noqa: BLE001  recorded: a config the kernel cannot run
                continue
            for g in groups:
                if torch.equal(g[0], out):
                    g[1].append(c)
                    break
            else:
                groups.append([out, [c]])
        # a parameter changes the bits when two configs that differ in it alone land in different groups
        member = {}
        for gi, g in enumerate(groups):
            for c in g[1]:
                member[tuple(sorted((k, str(v)) for k, v in c.items()))] = gi
        params = sorted(configs[0])
        affects = {}
        for k in params:
            buckets = {}
            for key, gi in member.items():
                rest = tuple(kv for kv in key if kv[0] != k)
                buckets.setdefault(rest, set()).add(gi)
            pairs = [b for b in buckets.values()]
            affects[k] = any(len(b) > 1 for b in pairs)
        key_ok = True
        if hasattr(tb, "order_key"):
            seen = {}
            for gi, g in enumerate(groups):
                for c in g[1]:
                    kk = tb.order_key(p, sig_env, c)
                    if kk in seen and seen[kk] != gi:
                        key_ok = False
                    seen.setdefault(kk, gi)
        row = {"case": name, "kind": tb.config_kind(p, sig_env), "configs": len(configs), "groups": len(groups),
               "changes_bits": sorted(k for k in params if affects[k]),
               "keeps_bits": sorted(k for k in params if not affects[k]),
               "order_key_consistent": key_ok}
        report.append(row)
        print(f"{name}: {len(configs)} configs, {len(groups)} bit groups; changes bits {row['changes_bits']}; "
              f"keeps bits {row['keeps_bits']}; order key consistent {key_ok}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
