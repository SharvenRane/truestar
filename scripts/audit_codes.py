"""Audit every Truestar code: raised in src, documented, referenced by a test, message quality.

Usage:
    PYTHONPATH=src python scripts/audit_codes.py              # markdown table on stdout
    PYTHONPATH=src python scripts/audit_codes.py --json out.json
    PYTHONPATH=src python scripts/audit_codes.py --worst 30   # lowest scoring raise sites

It reads the source with ``ast`` and runs nothing. For every code in ``truestar.codes.CODES``:

* raised: a string constant in src (not a docstring, not ``codes.py``) holds the code, either as
  the whole string (``Issue("TS010", msg)``, ``code = "TS1080"``) or inside message text
  (``f"[TS1021] ..."``). A class with ``code = "TSxxxx"`` counts each ``raise Class(...)`` as a
  site; a module name bound to a code (``CODE = "TS022"``) counts each call that passes it.
* documented: the code appears in docs/AGENT_INTERFACE.md.
* tested: a string constant in tests/ holds the code. This is static: it shows that a test names
  the code (usually an assertion on it), not that the test ran the raising line.
* message: the text next to the code at each site, with ``{expr}`` for interpolated values.
  ``cause`` is true when it states what is wrong (a negation, an unknown or missing thing, a
  comparison, an interpolated value); ``action`` is true when it tells the user what to do (an
  imperative verb such as use, pass, install, a ``--flag``, a docs/ pointer). INFO codes need no
  action. Both are word heuristics; read the worst list before acting on it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "truestar"
TESTS = ROOT / "tests"
AGENT_DOC = ROOT / "docs" / "AGENT_INTERFACE.md"
CODE_RE = re.compile(r"\bTS\d{3,4}\b")

ACTION_RE = re.compile(
    r"(?:^|[:;.,(]\s*|\b(?:then|or|and|to)\s+)"
    r"(use|pass|set|install|run|rerun|re-run|add|remove|drop|try|check|see|call|give|choose|pick|"
    r"upgrade|downgrade|reduce|increase|lower|raise|enable|disable|convert|cast|export|re-export|"
    r"rebuild|register|provide|specify|define|record|re-record|fix|look|move|split|replace|load|"
    r"select|write|regenerate|rename|keep|make|switch|update|declare|compile|wrap|restart|report|"
    r"file|open|start|delete|request|ask|supply|attach|build|annotate|mark|lift|allow|install)\b",
    re.I)
ACTION_HINT_RE = re.compile(r"--[a-z][\w-]*|docs/\w|pip install|\bpython -m\b|\btruestar \w+", re.I)
CAUSE_RE = re.compile(
    r"\b(not|no|cannot|can't|does not|did not|unknown|missing|failed|fails|invalid|unsupported|"
    r"differs?|different|mismatch|exceeds?|over|under|above|below|only|without|refused|expected|"
    r"must|too|neither|nor|wrong|unavailable|empty|none|disagree\w*|diverge\w*|overflow\w*|nan|"
    r"inf|non finite|because|but|instead|out of|outside|beyond|lost|changed)\b", re.I)


@dataclass
class Site:
    file: str
    line: int
    kind: str              # literal | text | class | alias
    message: str = ""
    dynamic: bool = False  # message is built elsewhere (a variable), not visible here
    cause: bool = False
    action: bool = False
    score: int = 0


@dataclass
class CodeRow:
    code: str
    severity: str
    slug: str
    meaning: str
    sites: List[Site] = field(default_factory=list)
    documented: bool = False
    tests: List[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------ AST helpers


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    par = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            par[child] = node
    return par


def _is_docstring(node: ast.AST, par) -> bool:
    p = par.get(node)
    return isinstance(p, ast.Expr)


def _render(node: Optional[ast.AST]) -> Optional[str]:
    """Readable text of a message expression, or None when it is not a string expression."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                out.append("{" + ast.unparse(v.value) + "}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        left, right = _render(node.left), _render(node.right)
        if left is None and right is None:
            return None
        return (left or "{" + ast.unparse(node.left) + "}") + (
            right or "{" + ast.unparse(node.right) + "}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "format":
        return _render(node.func.value)
    return None


MESSAGE_KW = {"message", "msg", "reason", "detail", "details", "why", "text", "note", "hint"}


def _resolve_name(name: str, at: ast.AST, par) -> Optional[str]:
    """Text assigned to a local name before ``at`` in its function (``msg = ...; msg += ...``)."""
    fn = _enclosing(at, par, (ast.FunctionDef, ast.AsyncFunctionDef))
    if fn is None:
        return None
    parts = []
    for node in ast.walk(fn):
        if getattr(node, "lineno", 1 << 30) > at.lineno:
            continue
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            text = _render(node.value)
            if text is not None:
                parts.append((node.lineno, text, True))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name:
            text = _render(node.value)
            if text is not None:
                parts.append((node.lineno, text, False))
    if not parts:
        return None
    parts.sort()
    # several plain assignments are branches; keep the last one and the additions after it
    start = max(i for i, p in enumerate(parts) if p[2]) if any(p[2] for p in parts) else 0
    return "".join(p[1] for p in parts[start:])


def _call_message(call: ast.Call, skip: Optional[ast.AST], par=None) -> tuple:
    """(message text, dynamic) from a call's positional arguments and message keywords."""
    parts, dynamic = [], False
    for a in list(call.args) + [k.value for k in call.keywords if k.arg in MESSAGE_KW]:
        if a is skip:
            continue
        text = _render(a)
        if text is None and isinstance(a, ast.Name) and par is not None:
            text = _resolve_name(a.id, call, par)
        if text is not None and not CODE_RE.fullmatch(text.strip()):
            parts.append(text)
        elif isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "str":
            parts.append("{" + ast.unparse(a) + "}")  # forwards an exception's own text
        elif isinstance(a, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            dynamic = True
    return " ".join(parts), dynamic and not parts


def _enclosing(node, par, types):
    while node is not None and not isinstance(node, types):
        node = par.get(node)
    return node


# ----------------------------------------------------------------------------------- scanning


def _scan_src(rows: Dict[str, CodeRow]) -> None:
    class_codes: Dict[str, str] = {}   # exception class name -> code
    init_classes = set()                # classes whose __init__ builds the message
    trees = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "codes.py" and path.parent == SRC:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:
            continue
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for st in node.body:
                    if isinstance(st, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "code" for t in st.targets):
                        v = st.value
                        if isinstance(v, ast.Constant) and isinstance(v.value, str) \
                                and v.value in rows:
                            class_codes[node.name] = v.value
                if any(isinstance(st, ast.FunctionDef) and st.name == "__init__"
                       for st in node.body):
                    init_classes.add(node.name)

    for path, tree in trees.items():
        rel = path.relative_to(ROOT).as_posix()
        par = _parents(tree)
        aliases: Dict[str, str] = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if _is_docstring(node, par):
                continue
            found = CODE_RE.findall(node.value)
            if not found:
                continue
            whole = CODE_RE.fullmatch(node.value.strip())
            parent = par.get(node)
            for code in set(found):
                if code not in rows:
                    continue
                if whole:
                    # bound to a name at module or class level: an alias or an exception class
                    if isinstance(parent, ast.Assign) and parent.value is node:
                        tgt = parent.targets[0]
                        owner = par.get(parent)
                        if isinstance(tgt, ast.Name) and isinstance(owner, ast.ClassDef) \
                                and tgt.id == "code":
                            continue  # counted through its raise sites
                        if isinstance(tgt, ast.Name) and isinstance(owner, ast.Module):
                            aliases[tgt.id] = code
                            continue
                    call = parent if isinstance(parent, ast.Call) else (
                        par.get(parent) if isinstance(parent, ast.keyword) else None)
                    if isinstance(parent, ast.Compare) or (
                            isinstance(parent, (ast.Tuple, ast.List, ast.Set))
                            and isinstance(par.get(parent), ast.Compare)):
                        continue  # a test of a code (code == "TS020"), not a raise
                    site = Site(rel, node.lineno, "literal")
                    if isinstance(parent, ast.Dict) and node in parent.keys:
                        # {"TS306": f"..."}: a code to message table
                        val = parent.values[parent.keys.index(node)]
                        text = _render(val)
                        site.message, site.dynamic = (text or ""), text is None
                    elif call is not None:
                        site.message, site.dynamic = _call_message(call, node, par)
                    elif isinstance(parent, ast.Tuple) and isinstance(par.get(parent), ast.Assign)                             and isinstance(par[parent].targets[0], ast.Tuple):
                        site.dynamic = True  # verdict, code = "label", "TS1099": message elsewhere
                    elif isinstance(parent, (ast.Tuple, ast.List)):
                        # ("TS905", f"...") pairs and return code, message
                        fake = ast.Call(func=ast.Name("t"), args=list(parent.elts), keywords=[])
                        site.message, site.dynamic = _call_message(fake, node, par)
                        site.dynamic = site.dynamic or not site.message
                    elif isinstance(parent, ast.Dict):
                        # {"code": "TS742", "message": f"..."}
                        vals = [v for k, v in zip(parent.keys, parent.values)
                                if isinstance(k, ast.Constant) and k.value in MESSAGE_KW]
                        fake = ast.Call(func=ast.Name("d"), args=vals, keywords=[])
                        site.message, site.dynamic = _call_message(fake, node, par)
                        site.dynamic = site.dynamic or not site.message
                    else:
                        site.dynamic = True
                    rows[code].sites.append(site)
                else:
                    host = node
                    if isinstance(parent, ast.JoinedStr):
                        host = parent
                    kw = par.get(host)
                    if isinstance(kw, ast.keyword) and kw.arg in ("help", "description", "epilog"):
                        continue  # argparse help text names the code; it raises nothing
                    text = _render(host) or node.value
                    rows[code].sites.append(Site(rel, node.lineno, "text", text))
        # alias uses: CODE passed to a call, or an exception class's code = CODE
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for a in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(a, ast.Name) and a.id in aliases:
                        msg, dyn = _call_message(node, a, par)
                        rows[aliases[a.id]].sites.append(
                            Site(rel, node.lineno, "alias", msg, dyn))
            if isinstance(node, ast.ClassDef):
                for st in node.body:
                    if isinstance(st, ast.Assign) and isinstance(st.value, ast.Name) \
                            and st.value.id in aliases and any(
                                isinstance(t, ast.Name) and t.id == "code" for t in st.targets):
                        class_codes[node.name] = aliases[st.value.id]

    # exceptions with no code of their own that a handler turns into one:
    #   except ModelFileError as exc: failure = ("TS020", str(exc), {})
    defined = {n.name for t in trees.values() for n in ast.walk(t) if isinstance(n, ast.ClassDef)}
    caught_all: Dict[str, list] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            names = [t.id if isinstance(t, ast.Name) else getattr(t, "attr", None) for t in types]
            found = [c.value.strip() for b in node.body for c in ast.walk(b)
                     if isinstance(c, ast.Constant) and isinstance(c.value, str)
                     and CODE_RE.fullmatch(c.value.strip()) and c.value.strip() in rows]
            for n in names:
                if n in defined and n not in class_codes:
                    caught_all.setdefault(n, []).append(found[0] if found else None)
    # credit a class to a code only when every handler that names a code names the same one and
    # coded handlers outnumber uncoded ones two to one; a general error (IRError) caught in many
    # places, mostly without a code, is not credited to the one place that adds one
    caught = {}
    for n, cs in caught_all.items():
        coded = [c for c in cs if c is not None]
        if len(set(coded)) == 1 and len(coded) >= 2 * (len(cs) - len(coded)):
            caught[n] = coded[0]
    for path, tree in trees.items():
        rel = path.relative_to(ROOT).as_posix()
        par = _parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                f = node.exc.func
                name = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else None)
                if name in caught:
                    msg, dyn = _call_message(node.exc, None, par)
                    if not CODE_RE.search(msg):
                        rows[caught[name]].sites.append(Site(rel, node.lineno, "caught", msg, dyn))
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                f = node.exc.func
                name = f.id if isinstance(f, ast.Name) else (
                    f.attr if isinstance(f, ast.Attribute) else None)
                if name in class_codes:
                    msg, dyn = _call_message(node.exc, None, par)
                    code = class_codes[name]
                    if CODE_RE.search(msg) and not any(
                            isinstance(a, ast.Name) for a in node.exc.args):
                        continue  # the message names its code itself: counted as a text site
                    if name in init_classes:
                        msg, dyn = "", True  # composed in __init__; read the class to grade it
                    rows[code].sites.append(Site(rel, node.lineno, "class", msg, dyn))


def _scan_tests(rows: Dict[str, CodeRow]) -> None:
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:
            continue
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.update(c for c in CODE_RE.findall(node.value) if c in rows)
        for code in seen:
            rows[code].tests.append(path.name)


def _grade(row: CodeRow) -> None:
    from truestar.codes import INFO
    for s in row.sites:
        text = CODE_RE.sub("", s.message).strip(" []:")
        words = re.findall(r"[A-Za-z]+", re.sub(r"\{[^}]*\}", " ", text))
        if not s.dynamic and not words and "{" in text:
            s.kind += "+wrapper"
        if s.dynamic or not words:
            s.cause = s.action = False
            s.score = -1  # nothing visible to grade here
            continue
        if "{" in text and len(words) <= 3 and not CAUSE_RE.search(text):
            s.kind += "+wrapper"  # frames another error's text, e.g. "error [TS001]: {exc}"
            s.score = -1
            continue
        s.cause = bool(CAUSE_RE.search(text)) or ("{" in text and len(words) >= 3)
        s.action = row.severity == INFO or bool(ACTION_RE.search(text) or ACTION_HINT_RE.search(text))
        s.score = int(s.cause) + int(s.action) + (1 if len(words) >= 6 else 0)


def build() -> List[CodeRow]:
    sys.path.insert(0, str(ROOT / "src"))
    from truestar.codes import CODES
    rows = {c: CodeRow(c, v[0], v[1], v[2]) for c, v in CODES.items()}
    doc = AGENT_DOC.read_text(encoding="utf-8")
    for c, r in rows.items():
        r.documented = re.search(rf"\b{c}\b", doc) is not None
    _scan_src(rows)
    for r in rows.values():
        uniq = {}
        for st in r.sites:
            uniq.setdefault((st.file, st.line, st.message), st)
        r.sites = list(uniq.values())
    _scan_tests(rows)
    for r in rows.values():
        _grade(r)
    return sorted(rows.values(), key=lambda r: int(r.code[2:]))


def _markdown(rows: List[CodeRow]) -> str:
    out = ["| code | severity | slug | raised (sites) | documented | tests | sites with cause | "
           "sites with action |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        graded = [s for s in r.sites if s.score >= 0]
        cause = f"{sum(s.cause for s in graded)}/{len(graded)}" if graded else "n/a"
        act = f"{sum(s.action for s in graded)}/{len(graded)}" if graded else "n/a"
        out.append(f"| {r.code} | {r.severity} | {r.slug} | {len(r.sites) or 'DEAD'} | "
                   f"{'yes' if r.documented else 'NO'} | {len(r.tests) or 'NONE'} | {cause} | {act} |")
    n = len(rows)
    dead = [r.code for r in rows if not r.sites]
    undoc = [r.code for r in rows if not r.documented]
    untested = [r.code for r in rows if not r.tests]
    sites = [s for r in rows for s in r.sites]
    graded = [s for s in sites if s.score >= 0]
    out += ["", f"codes: {n}", f"never raised in src: {len(dead)} {' '.join(dead)}",
            f"not in docs/AGENT_INTERFACE.md: {len(undoc)} {' '.join(undoc)}",
            f"named by no test: {len(untested)} {' '.join(untested)}",
            f"raise sites: {len(sites)}, with visible message: {len(graded)}, "
            f"naming a cause: {sum(s.cause for s in graded)}, "
            f"naming an action: {sum(s.action for s in graded)}, "
            f"both: {sum(s.cause and s.action for s in graded)}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="write the full table with every site to this file")
    ap.add_argument("--worst", type=int, default=0, help="print the N lowest scoring sites")
    args = ap.parse_args(argv)
    rows = build()
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in rows], indent=1),
                                   encoding="utf-8")
    if args.worst:
        sev = {"error": 0, "warning": 1, "info": 2}
        sites = [(s, r) for r in rows for s in r.sites if s.score >= 0]
        sites.sort(key=lambda p: (p[0].score, sev[p[1].severity], len(p[0].message)))
        for s, r in sites[:args.worst]:
            print(f"{r.code} {r.severity} {s.file}:{s.line} score={s.score} "
                  f"cause={s.cause} action={s.action}\n    {s.message!r}")
        return 0
    print(_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
