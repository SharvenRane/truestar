"""`truestar fingerprint`: run the cross chip fingerprint, or compare two (docs/CROSS_CHIP.md).

    truestar fingerprint --json fp.json [--cases mlp,transformer] [--dtypes float32,float16]
                         [--batches 1,7] [--arms truestar_det,eager] [--data-dir D] [--fresh-cache]
    truestar fingerprint compare a.json b.json [--json diff.json]

Exit codes. Run: 0 every claimed (deterministic) row was equal run to run, 1 one was not (TS741),
2 usage error or no CUDA GPU. Compare: 0 every deterministic row present in both files is bitwise
equal, 1 one differs (TS743), 2 a file is not a fingerprint (TS749).
"""

from __future__ import annotations

import json
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def add_parser(sub) -> None:
    p = sub.add_parser("fingerprint", help="bit level hashes of compiled and eager results on this GPU, or "
                                           "'compare A B' to diff two")
    p.add_argument("action", nargs="?", default="run", choices=("run", "compare"))
    p.add_argument("files", nargs="*", help="compare: the two fingerprint JSON files")
    p.add_argument("--json", dest="json_path", help="write the fingerprint (or the comparison) here")
    p.add_argument("--cases", help="comma list (default: every case)")
    p.add_argument("--dtypes", help="comma list of float32, tf32, float16, bfloat16 (default: all)")
    p.add_argument("--batches", help="comma list of batch sizes (default: 1,7,64,256)")
    p.add_argument("--arms", help="comma list of truestar_det, truestar_default, eager, inductor (default: those four), "
                        "truestar_portable (opt in: portable tensor cores)")
    p.add_argument("--data-dir", help="where the weights and inputs files live (default: in the temp directory)")
    p.add_argument("--fresh-cache", action="store_true",
                   help="compile and tune into an empty kernel cache (no config from an earlier process)")
    p.add_argument("--format", choices=("text", "agent"), default="text")


def _envelope(command, status, exit_code, issues, body):
    from truestar import __version__

    return {"schema_version": 1, "tool": "truestar", "command": command, "truestar_version": __version__,
            "status": status, "exit_code": exit_code, "codes": sorted({i["code"] for i in issues}),
            "issues": issues, **body}


def _usage(a, code, msg) -> int:
    if a.format == "agent":
        print(json.dumps(_envelope(f"fingerprint {a.action}", "error", EXIT_USAGE,
                                   [{"code": code, "message": msg}], {"next_actions": []})))
    else:
        print(f"truestar fingerprint: {msg}", file=sys.stderr)
    return EXIT_USAGE


def _split(v):
    return None if not v else [s.strip() for s in v.split(",") if s.strip()]


def run(a) -> int:
    if a.action == "compare":
        return _compare(a)
    if a.files:
        return _usage(a, "TS021", f"unexpected arguments {' '.join(a.files)} (did you mean "
                                  f"'truestar fingerprint compare A B'?)")
    from .data import DataMismatch
    from .run import run as run_fp, to_text

    try:
        batches = [int(b) for b in _split(a.batches)] if a.batches else None
    except ValueError:
        return _usage(a, "TS021", f"--batches {a.batches!r}: write integers, for example 1,7,64")
    log = None if a.format == "agent" else (lambda s: print(s, file=sys.stderr, flush=True))
    try:
        doc = run_fp(_split(a.cases), _split(a.dtypes), batches, _split(a.arms), a.data_dir, a.fresh_cache, log)
    except DataMismatch as exc:
        return _usage(a, "TS740", str(exc))
    except (ValueError, RuntimeError) as exc:
        return _usage(a, "TS021", str(exc))
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, default=str)
    code = EXIT_FINDING if any(i["code"] == "TS741" for i in doc["issues"]) else EXIT_OK
    if a.format == "agent":
        body = {"fingerprint": doc, "next_actions": [
            {"kind": "compare", "description": "Run the same command on another GPU, then compare the two files.",
             "command": f"truestar fingerprint compare {a.json_path or 'A.json'} OTHER.json"}]}
        print(json.dumps(_envelope("fingerprint", "ok" if code == EXIT_OK else "finding", code, doc["issues"], body),
                         default=str))
    else:
        print(to_text(doc))
        if a.json_path:
            print(f"wrote {a.json_path}")
    return code


def _compare(a) -> int:
    from .compare import FingerprintFileError, compare_files

    if len(a.files) != 2:
        return _usage(a, "TS021", f"compare takes exactly two fingerprint JSON files, got {len(a.files)}: "
                                  "truestar fingerprint compare A.json B.json")
    try:
        cmp = compare_files(*a.files)
    except FingerprintFileError as exc:
        return _usage(a, "TS749", str(exc))
    code = EXIT_OK if cmp.claim_ok else EXIT_FINDING
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(cmp.as_dict(), f, indent=1, default=str)
    if a.format == "agent":
        print(json.dumps(_envelope("fingerprint compare", "ok" if code == EXIT_OK else "finding", code, cmp.issues,
                                   {"comparison": cmp.as_dict(), "next_actions": []}), default=str))
    else:
        print(cmp.to_text())
    return code
