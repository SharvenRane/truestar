"""Child process of check_docs_snippets.py: do the truestar names and keywords in a snippet exist?

    PYTHONPATH=src python scripts/_docs_resolve.py snippet.py FIRST_LINE

Reads the snippet, imports what its truestar imports name, and resolves every attribute chain that
starts at one of those names (through modules, then one step into the first non module) and every
call to a resolved callable against its signature (keyword names, and the positional count when the
callee takes no *args). Prints one JSON document: {"syntax": message} or {"problems": [...]}, where
each problem has a line and a message. Only problems at or after FIRST_LINE are printed, so earlier
blocks given for context are not reported twice.
"""

import ast
import importlib
import inspect
import json
import sys

MISSING = object()


def import_path(path):
    try:
        return importlib.import_module(path)
    except ImportError:
        return MISSING


def resolve_from(base, name):
    """`from base import name`: an attribute of the module, or its submodule."""
    mod = import_path(base)
    if mod is MISSING:
        return MISSING
    obj = getattr(mod, name, MISSING)
    if obj is MISSING:
        obj = import_path(base + "." + name)
    return obj


def chain(node):
    """Name.attr.attr as a list of strings, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


class Resolver:
    def __init__(self, tree):
        self.names = {}          # local name -> object
        self.problems = []
        self.tree = tree

    def problem(self, node, msg):
        self.problems.append({"line": getattr(node, "lineno", 0), "message": msg})

    def imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if not a.name.split(".")[0] == "truestar":
                        continue
                    mod = import_path(a.name)
                    if mod is MISSING:
                        self.problem(node, f"import {a.name}: no such module")
                        continue
                    if a.asname:
                        self.names[a.asname] = mod
                    else:
                        self.names["truestar"] = import_path("truestar")
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.split(".")[0] != "truestar":
                    continue
                if import_path(node.module) is MISSING:
                    self.problem(node, f"from {node.module} import ...: no such module")
                    continue
                for a in node.names:
                    obj = resolve_from(node.module, a.name)
                    if obj is MISSING:
                        self.problem(node, f"from {node.module} import {a.name}: {node.module} has no {a.name}")
                    else:
                        self.names[a.asname or a.name] = obj

    def rebound(self):
        """Import names the snippet later assigns to are not followed (they are something else then)."""
        out = set()
        for node in ast.walk(self.tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
                targets = [node.target]
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out.add(node.name)
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        return out

    def lookup(self, node):
        """The object an expression names, following module attributes; MISSING if unknown."""
        parts = chain(node)
        if not parts or parts[0] not in self.names:
            return MISSING
        obj, path = self.names[parts[0]], parts[0]
        for attr in parts[1:]:
            if not inspect.ismodule(obj):
                return MISSING if not hasattr(obj, attr) else getattr(obj, attr)
            nxt = getattr(obj, attr, MISSING)
            if nxt is MISSING:
                nxt = import_path(obj.__name__ + "." + attr)
            if nxt is MISSING:
                return MISSING
            obj, path = nxt, path + "." + attr
        return obj

    def attributes(self):
        seen = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = chain(node)
            if not parts or parts[0] not in self.names or tuple(parts) in seen:
                continue
            seen.add(tuple(parts))
            obj = self.names[parts[0]]
            for k, attr in enumerate(parts[1:], 1):
                was_module = inspect.ismodule(obj)
                nxt = getattr(obj, attr, MISSING)
                if nxt is MISSING and was_module:
                    nxt = import_path(obj.__name__ + "." + attr)
                if nxt is MISSING:
                    self.problem(node, f"{'.'.join(parts[:k])} has no attribute {attr}")
                    break
                obj = nxt
                if not was_module:
                    break

    def calls(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            fn = self.lookup(node.func)
            if fn is MISSING or not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                continue
            params = list(sig.parameters.values())
            kinds = {p.kind for p in params}
            name = ".".join(chain(node.func) or ["?"])
            if inspect.Parameter.VAR_KEYWORD not in kinds:
                allowed = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in allowed:
                        self.problem(node, f"{name}() takes no keyword {kw.arg!r} "
                                           f"(it takes {', '.join(sorted(allowed)) or 'none'})")
            if inspect.Parameter.VAR_POSITIONAL not in kinds and not any(
                    isinstance(a, ast.Starred) for a in node.args):
                npos = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
                if len(node.args) > npos:
                    self.problem(node, f"{name}() takes at most {npos} positional arguments, "
                                       f"the snippet passes {len(node.args)}")


def main():
    path, first = sys.argv[1], int(sys.argv[2])
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(json.dumps({"syntax": f"line {e.lineno}: {e.msg}"}))
        return 0
    r = Resolver(tree)
    r.imports()
    for name in r.rebound():
        r.names.pop(name, None)
    r.attributes()
    r.calls()
    probs = [p for p in r.problems if p["line"] >= first]
    uniq = []
    for p in probs:
        if p not in uniq:
            uniq.append(p)
    for p in uniq:
        p["line"] -= first - 1
    print(json.dumps({"problems": uniq}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
