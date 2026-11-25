"""Every error code appears once in the CODES literal.

A dict literal silently keeps the last of two equal keys, so a code reused by two features (as
TS185 nearly was, by the RL kernel set and the near tie gradient note) would lose one meaning with
no error. This reads the source, not the built dict.
"""
import ast
import collections
import pathlib

import truestar.codes as codes

SOURCE = pathlib.Path(codes.__file__).read_text(encoding="utf-8")


def _literal_keys():
    for node in ast.walk(ast.parse(SOURCE)):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == "CODES" and isinstance(value, ast.Dict):
            return [k.value for k in value.keys if isinstance(k, ast.Constant)]
    raise AssertionError("CODES dict literal not found in truestar/codes.py")


def test_every_code_is_defined_once():
    keys = _literal_keys()
    dupes = {k: n for k, n in collections.Counter(keys).items() if n > 1}
    assert not dupes, f"codes defined more than once: {dupes}"


def test_every_code_name_is_unique():
    names = [v[1] for v in codes.CODES.values()]
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    assert not dupes, f"code names used by more than one code: {dupes}"


def test_the_check_sees_a_duplicate():
    """Guard for the guard: a duplicate written into a copy of the source is found."""
    bad = SOURCE.replace('CODES: Dict[str, tuple] = {', 'CODES: Dict[str, tuple] = {\n    "TS180": (None, "x", ""),', 1)
    assert bad != SOURCE
    tree = ast.parse(bad)
    keys = [k.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "CODES"
            for k in n.value.keys if isinstance(k, ast.Constant)]
    assert collections.Counter(keys)["TS180"] == 2


def test_codes_are_three_or_four_digits_and_sort_by_number():
    """Four digit codes (TS1060 and up, 2026-11-24) are accepted; `truestar codes` lists them by number."""
    import re

    from truestar.agent import code_table
    keys = _literal_keys()
    assert all(re.fullmatch(r"TS\d{3,4}", k) for k in keys), [k for k in keys if not re.fullmatch(r"TS\d{3,4}", k)]
    nums = [int(k[2:]) for k in keys]
    assert len(set(nums)) == len(nums), "two codes with the same number (TS0123 and TS123 would be one)"
    table = [int(r["code"][2:]) for r in code_table()]
    assert table == sorted(table)
