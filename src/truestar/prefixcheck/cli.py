"""`truestar prefixcheck`: compare an engine's cached prefix run with its full prefill, from files.

    truestar prefixcheck full.pt cached.pt [--reference ref.pt] [--tolerance 1e-3] [--format agent]

Each file is ``.pt`` (a logits tensor, or a dict with ``logits`` and optionally ``layers``) or ``.npz``
(``logits`` and ``layers/<name>``), the logits at the same new tokens. Exit codes: 0 bitwise equal
(TS704), 1 they differ (TS702, and TS703 when the cached run is less accurate than the full prefill
against the reference), 2 usage error (a file missing or unreadable, shapes that differ).
"""

from __future__ import annotations

import json
import sys

EXIT_OK, EXIT_FINDING, EXIT_USAGE = 0, 1, 2


def add_parser(sub) -> None:
    p = sub.add_parser("prefixcheck", help="does a cached prefix run give the full prefill's bits (files from "
                                           "any engine)")
    p.add_argument("full", help="logits of the full prefill at the new tokens (.pt or .npz)")
    p.add_argument("cached", help="logits of the cached prefix run at the same tokens")
    p.add_argument("--reference", help="float64 reference logits, to tell rounding from a less accurate path")
    p.add_argument("--tolerance", type=float, default=None, help="TS703 extra relative error allowed")
    p.add_argument("--json", dest="json_path", help="also write the report as JSON here")
    p.add_argument("--format", choices=("text", "agent"), default="text")


def run(a) -> int:
    from truestar import __version__

    from . import DEFAULT_TOL, check_files
    try:
        rep = check_files(a.full, a.cached, a.reference, a.tolerance if a.tolerance is not None else DEFAULT_TOL)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if a.format == "agent":
            print(json.dumps({"schema_version": 1, "tool": "truestar", "command": "prefixcheck",
                              "truestar_version": __version__, "status": "error", "exit_code": EXIT_USAGE,
                              "codes": ["TS021"], "issues": [{"code": "TS021", "message": msg}],
                              "next_actions": []}))
        else:
            print(f"truestar prefixcheck: {msg}", file=sys.stderr)
        return EXIT_USAGE
    code = EXIT_OK if rep.bitwise else EXIT_FINDING
    body = rep.to_dict()
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=1)
    if a.format == "agent":
        print(json.dumps({"schema_version": 1, "tool": "truestar", "command": "prefixcheck",
                          "truestar_version": __version__, "status": "ok" if rep.bitwise else "finding",
                          "exit_code": code, "codes": sorted(set(rep.codes)), "issues": rep.issues,
                          "next_actions": [], **body}))
    else:
        print(rep.to_text())
    return code
