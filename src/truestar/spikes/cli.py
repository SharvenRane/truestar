"""`truestar replay RUN --step N`: the command line side of truestar.spikes.replay (docs/SPIKES.md).

Exit codes: 0 no spike at that step, 1 a verdict (numerics, data, optimizer) or a replay that does
not reproduce the log, 2 usage error (TS697: not a run directory, missing checkpoint or batch).
"""

from __future__ import annotations

import contextlib
import json
import sys


def add_parser(sub) -> None:
    r = sub.add_parser("replay", help="replay a training step from a spike watch run directory and say whether a "
                                      "spike is numerics, data or the optimizer (docs/SPIKES.md)")
    r.add_argument("run", help="run directory written by truestar.spikes.SpikeWatch")
    r.add_argument("--step", type=int, required=True, help="the step to explain")
    r.add_argument("--spec", default=None, help="spec file (default: the one recorded in the run)")
    r.add_argument("--json", dest="json_path", help="also write the agent JSON here")
    r.add_argument("--format", choices=("text", "agent"), default="text")


def run(a) -> int:
    from ..agent import dumps, error_to_agent

    agent = a.format == "agent"
    quiet = contextlib.redirect_stdout(sys.stderr) if agent else contextlib.nullcontext()
    failure = rep = None
    with quiet:
        from .replay import ReplayError, replay

        try:
            rep = replay(a.run, a.step, a.spec)
        except (ReplayError, FileNotFoundError, ValueError) as exc:
            failure = str(exc)
    if failure is not None:
        if agent:
            doc = error_to_agent("TS697", failure)
            doc["command"] = "replay"
            print(dumps(doc))
        else:
            print(f"error [TS697]: {failure}", file=sys.stderr)
        return 2
    doc = rep.to_agent(a.run)
    print(dumps(doc) if agent else rep.to_text())
    if a.json_path:
        with open(a.json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2, allow_nan=False))
    return 0 if rep.verdict == "none" else 1
