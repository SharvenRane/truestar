"""`truestar serve`: run the server, or check its guarantee.

    truestar serve [--preset tiny|bench] [--device cpu|cuda|reference] [--port 8000] [--chunk N]
                   [--no-prefix-cache] [--max-batch 32] [--max-rows 512]
    truestar serve --check [--device cpu|cuda] [--format agent] [--json out.json]

Exit codes of --check: 0 the guarantee holds (TS1400) and the float64 check agrees (TS1408), 1 a
finding (TS1401 or TS1402), 2 usage error or the executor is unavailable (TS1407).
"""

from __future__ import annotations

import json
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def add_parser(sub) -> None:
    p = sub.add_parser("serve", help="a small LLM server whose answers do not change with batching, prefix "
                                     "caching or chunked prefill (OpenAI compatible subset)")
    p.add_argument("--preset", default="tiny", choices=("tiny", "bench"), help="model built from a config in code")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "reference"),
                   help="compiled float32 on cpu or cuda, or the float64 reference interpreter")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-batch", type=int, default=32, help="running sequences at most")
    p.add_argument("--max-rows", type=int, default=512, help="tokens per step at most")
    p.add_argument("--chunk", type=int, default=None, help="chunked prefill: prompt tokens per step")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--check", action="store_true", help="check the guarantee and the float64 agreement, then exit")
    p.add_argument("--json", dest="json_path")
    p.add_argument("--format", choices=("text", "agent"), default="text")


def _agent(status, code, codes, issues, **body):
    from truestar import __version__

    print(json.dumps({"schema_version": 1, "tool": "truestar", "command": "serve", "truestar_version": __version__,
                      "status": status, "exit_code": code, "codes": codes, "issues": issues, "next_actions": [],
                      **body}))


def run(a) -> int:
    from .model import TinyLM, config

    if a.check:
        return _check(a)
    from .engine import Engine, StepRunner
    from .http import serve

    try:
        model = TinyLM(config(a.preset))
        runner = StepRunner(model, a.device)
    except Exception as exc:   # no GPU, no numba
        print(f"truestar serve: TS1407 the executor {a.device!r} is unavailable: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if runner.fallbacks:
        print(f"truestar serve: TS1406 the compiled step has {runner.fallbacks} fallbacks", file=sys.stderr)
    eng = Engine(model, runner=runner, max_batch=a.max_batch, chunk=a.chunk, prefix_cache=not a.no_prefix_cache,
                 max_rows=a.max_rows)
    eng.warmup()
    srv = serve(eng, a.host, a.port, model_name=f"truestar-{a.preset}")
    print(f"truestar serve: {a.preset} model on {a.device}, http://{a.host}:{srv.server_address[1]}/v1", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        srv.server_close()
    return EXIT_OK


def _check(a) -> int:
    from .check import run_check

    if a.device == "reference":
        print("truestar serve --check: --device must be cpu or cuda (the reference is the float64 side)",
              file=sys.stderr)
        return EXIT_USAGE
    try:
        rep = run_check(a.preset, a.device)
    except Exception as exc:
        msg = f"TS1407 the executor {a.device!r} is unavailable: {type(exc).__name__}: {exc}"
        if a.format == "agent":
            _agent("error", EXIT_USAGE, ["TS1407"], [{"code": "TS1407", "message": msg}])
        else:
            print(f"truestar serve --check: {msg}", file=sys.stderr)
        return EXIT_USAGE
    bad = "TS1401" in rep["codes"] or "TS1402" in rep["codes"]
    if rep["fallbacks"]:
        rep["codes"].append("TS1406")
    code = EXIT_FINDING if bad else EXIT_OK
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    if a.format == "agent":
        issues = [{"code": c, "message": m} for c, m in _messages(rep)]
        _agent("finding" if bad else "ok", code, sorted(set(rep["codes"])), issues, report=rep)
        return code
    for r in rep["rows"]:
        mark = "bitwise" if r["bitwise"] else f"DIFFERS from token {r['first_difference']}"
        print(f"{r['scenario']:44} {r['request']:8} {r['tokens']} tokens, {r['cached_tokens']:3} cached: {mark}")
    f = rep.get("float64")
    if f:
        print(f"float64 reference: greedy tokens {'equal' if f['tokens_equal'] else 'differ at ' + str(f['first_difference'])}"
              f", float32 logprob error {f['float32_logprob_max_error']:.2e}; reference vs no cache: tokens "
              f"{'equal' if f['reference_vs_no_cache_tokens_equal'] else 'DIFFER'}, logprob error "
              f"{f['reference_vs_no_cache_logprob_error']:.1e}")
    for c, m in _messages(rep):
        print(f"{c}: {m}")
    return code


def _messages(rep):
    out = []
    if "TS1400" in rep["codes"]:
        out.append(("TS1400", f"every scenario gives the alone run's tokens and logprobs bit for bit ({rep['executor']})"))
    if "TS1401" in rep["codes"]:
        bad = [r for r in rep["rows"] if not r["bitwise"]]
        out.append(("TS1401", f"{len(bad)} scenario runs differ, first: {bad[0]['scenario']} ({bad[0]['request']}) "
                              f"from token {bad[0]['first_difference']}"))
    if "TS1402" in rep["codes"]:
        out.append(("TS1402", "the float32 server disagrees with the float64 reference beyond a near tie"))
    if "TS1408" in rep["codes"]:
        out.append(("TS1408", "greedy tokens equal the float64 reference server, which equals a forward with no cache"))
    if "TS1406" in rep["codes"]:
        out.append(("TS1406", f"the compiled step has {rep['fallbacks']} fallbacks"))
    return out
