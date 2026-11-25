"""scripts/check_docs_snippets.py: it finds every Python block and example command, and grades runs.

Fast: one run is a two line snippet in a fresh process; nothing imports torch in a child.
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    path = os.path.join(ROOT, "scripts", "check_docs_snippets.py")
    spec = importlib.util.spec_from_file_location("check_docs_snippets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = """# Title

```python
x = 1
```

```
truestar check model.py
```

* a list item with an indented block:

  ```py
  y = 2
      z = 3
  ```

````python
s = '''
```
'''
````

```json
{"a": 1}
```
"""


def test_extracts_python_blocks_only_with_their_line_and_indent():
    c = load()
    blocks = c.extract_blocks(MD)
    assert [line for line, _ in blocks] == [4, 14, 19]
    assert blocks[0][1] == "x = 1\n"
    assert blocks[1][1] == "y = 2\n    z = 3\n"       # the fence's indent is removed, the rest kept
    assert "```" in blocks[2][1]                       # a longer fence holds a shorter one


def test_docstring_commands_inline_wrapped_and_env_prefix():
    c = load()
    src = '''"""`truestar parallel examples/m.py --layout tp=2
--kernels rl`: wrapped over two lines. A mention of `truestar kernel` is not a command.

    truestar record examples/m.py -o before.tsrec      # comment
    TRUESTAR_X=1 truestar diff before.tsrec --replay
Run: ``PYTHONPATH=src python examples/m.py``
"""
'''
    cmds = c.docstring_commands(src)
    assert cmds == [
        ["truestar", "parallel", "examples/m.py", "--layout", "tp=2", "--kernels", "rl"],
        ["truestar", "record", "examples/m.py", "-o", "before.tsrec"],
        ["TRUESTAR_X=1", "truestar", "diff", "before.tsrec", "--replay"],
        ["PYTHONPATH=src", "python", "examples/m.py"],
    ]
    env, argv = c.to_argv(cmds[2])
    assert c.to_argv(cmds[3])[0] == {}                 # PYTHONPATH=src is the checkout, set already
    assert env == {"TRUESTAR_X": "1"} and argv[1:] == ["-m", "truestar", "diff", "before.tsrec",
                                                         "--replay"]


def test_skip_table_entries_point_at_real_text():
    c = load()
    items = c.collect()
    for path, needle, category, reason in c.SKIP:
        assert reason and category, (path, needle)
        assert any(it["rel"] == path and needle in it["text"] for it in items), (path, needle)


def test_run_grades_pass_fail_and_reruns_after_earlier_blocks():
    c = load()
    ok = c.run_item({"id": "a:1", "kind": "doc", "rel": "a", "text": "", "code": "print(1)\n",
                     "before": []})
    assert ok["status"] == "pass", ok
    bad = c.run_item({"id": "a:2", "kind": "doc", "rel": "a", "text": "",
                      "code": "raise SystemExit('boom')\n", "before": []})
    assert bad["status"] == "fail" and "boom" in bad["tail"]
    later = c.run_item({"id": "a:3", "kind": "doc", "rel": "a", "text": "", "code": "print(v)\n",
                        "before": ["v = 5\n"]})
    assert later["status"] == "pass" and later["with_earlier_blocks"]


def test_a_sketch_with_a_wrong_truestar_name_fails_and_names_it():
    c = load()
    code = ("import truestar\nimport truestar.fix\nfrom truestar.export import export, laod\n"
            "truestar.fix.fixx(model)\ntruestar.check(model, inputs, backendz=['cpu'])\n")
    r = c.run_item({"id": "a:4", "kind": "doc", "rel": "a", "text": code, "code": code, "before": []})
    assert r["status"] == "fail", r
    for needle in ("has no laod", "has no attribute fixx", "no keyword 'backendz'"):
        assert needle in r["reason"], r["reason"]
    good = "import truestar\nreport = truestar.check(model, inputs)\n"
    r = c.run_item({"id": "a:5", "kind": "doc", "rel": "a", "text": good, "code": good, "before": []})
    assert r["status"] == "fragment" and r["missing"] == "model", r


def test_a_block_that_forgets_an_import_fails():
    c = load()
    code = "import truestar\ny = torch.zeros(3)\n"
    r = c.run_item({"id": "a:6", "kind": "doc", "rel": "a", "text": code, "code": code, "before": []})
    assert r["status"] == "fail" and "missing import: 'torch'" in r["reason"], r
