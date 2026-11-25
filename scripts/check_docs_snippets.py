"""Docs that run: every Python block in the READMEs and docs/*.md, and every example, run on a CPU.

    PYTHONPATH=src python scripts/check_docs_snippets.py [--only TEXT] [--json out.json]
        [--timeout 120] [--list]

What it runs, one item at a time, each in a fresh process from a temporary directory (with a copy of
examples/, so relative example paths resolve and nothing lands in the checkout):

* every fenced ```python (or ```py) block of README_DRAFT.md, README.md and docs/*.md. A block that
  stops on a NameError is run again after the blocks before it in the same file, since docs often
  build one block on the last; when that passes the block counts as running.
* every example in examples/: the shell commands its docstring lists (``truestar ...`` and
  ``python ...`` lines, with ``NAME=value`` prefixes as environment), in order, in one directory so
  ``record`` then ``diff`` or ``export`` then ``run`` work. An exit code of 0 or 1 without a Python
  traceback counts as running, since 1 is how a check reports the finding an example demonstrates.
* the code cells of examples/notebooks/*.ipynb, as one script (``%%writefile`` becomes a file write).

The GPU is hidden with ``CUDA_VISIBLE_DEVICES=-1``, threads are capped, and each item has a timeout.
(An empty ``CUDA_VISIBLE_DEVICES=`` is not enough: with torch 2.11 on Windows the CUDA runtime still
sees the GPU, ``torch.cuda.is_available()`` is True and ``torch.cuda.device_count()`` is 0.)

Most doc blocks are sketches that use the reader's own ``model`` or ``x``. A block that fails is
checked statically (scripts/_docs_resolve.py): every truestar import, attribute and keyword
argument in it must exist. Then the status is ``fragment`` (it stopped on a NameError for a name
the reader supplies) or ``gpu`` (it stopped because CUDA is hidden); if a truestar name does not
resolve, or the error is anything else, the status is ``fail`` with the cause.

Items that cannot run here at all (schematic pseudo code, other hardware, timing runs) are listed in
SKIP with the reason; each entry matches a doc (or example) path and a piece of text from the block
or command, so it survives edits around it. Exit code: 0 when no item fails or times out, else 1.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
EXAMPLES = os.path.join(ROOT, "examples")

# (path relative to ROOT, text that appears in the block or command, category, reason)
SKIP = [
    ("docs/FRONTEND.md", "if h.sum() > 0: h = relu(h)", "schematic",
     "pseudo code of a benchmark case; the runnable version is scripts/frontend_vs_torch_compile.py"),
    ("docs/FRONTEND.md", "h = zeros(B, 32)", "schematic",
     "pseudo code of a benchmark case; the runnable version is scripts/frontend_vs_torch_compile.py"),
    ("docs/FRONTEND.md", "h = h @ w              # bug: [B, 32] @ [64, 32]", "schematic",
     "pseudo code of a benchmark case; the runnable version is scripts/frontend_vs_torch_compile.py"),
    ("docs/RANDOM.md", 'noise = torch.randn(1, 64, 2048, device="cuda")', "contrast",
     "plain PyTorch on CUDA, shown for contrast"),
    ("docs/RANDOM.md", "key = jax.random.key(42)", "contrast", "JAX code, shown for contrast"),
    ("docs/LLM.md", "k_cache: Tensor[L, \"B\", HKV", "schematic",
     "a decode program outline with `...` for its other arguments and layers"),
    ("docs/DIAGNOSTICS.md", 'dist.init_process_group("nccl")', "hardware",
     "a multi rank NCCL launch (torchrun on several GPUs)"),
    ("README_DRAFT.md", "def get_model():      # an nn.Module", "schematic",
     "the three function names of a model file, bodies left out"),
    ("docs/KERNEL_CHECK.md", "@truestar.verified_autotune(configs=[...]", "schematic",
     "a Triton kernel outline with `...` for arguments; needs CUDA and Triton"),
    ("examples/fix_demo.py", "--backends cuda", "gpu", "the simulated bad kernel is a CUDA addmm"),
    ("examples/rl_kernels_pair.py", "--device cuda", "gpu", "the RL kernel set runs on CUDA (Triton)"),
    ("examples/nn_mlp.py", "--bench", "benchmark", "a timing run; not run on a shared machine"),
    ("examples/nn_cnn.py", "--bench", "benchmark", "a timing run; not run on a shared machine"),
    ("examples/nn_gpt.py", "--bench", "benchmark", "a timing run; not run on a shared machine"),
]

_MLP = ("import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Sequential(torch.nn.Linear(8, 16), "
        "torch.nn.GELU(), torch.nn.Linear(16, 4)).eval()\n")

# Small CPU inputs for the user facing sketches, so they execute and not only resolve:
# (path, text in the block, code run before the block). A block that then runs is "setup"; one
# that fails is a real failure with plausible inputs.
SETUP = [
    ("README_DRAFT.md", "report = truestar.check(model, inputs)", _MLP + "inputs = torch.randn(3, 8)\n"),
    ("README.md", "report = truestar.check(model, inputs)", _MLP + "inputs = torch.randn(3, 8)\n"),
    ("docs/GRAD_CHECK.md", "truestar.grad_check(model, inputs", _MLP + "inputs = torch.randn(3, 8)\n"),
    ("docs/INVARIANCE.md", "invariance(model, inputs", _MLP + "inputs = torch.randn(6, 8)\n"),
    ("docs/RANGE.md", "range_check(model, inputs", _MLP + "inputs = torch.randn(3, 8)\n"),
    ("docs/FIX.md", "truestar.fix.fix(model, x)",     # a wrong float32 CPU addmm, so there is a fix
     _MLP + "x = torch.randn(3, 8)\nfrom truestar.testing import OpFault\n"
            "OpFault('aten.addmm.default', 'cpu', amount=0.05, skip_float64=True).__enter__()\n"),
    ("docs/PORT_CHECK.md", "port_check(reference_model, candidate_model, inputs)",
     _MLP + "import copy\nreference_model, candidate_model = model, copy.deepcopy(model)\n"
            "inputs = torch.randn(3, 8)\n"),
    ("docs/EXPORT.md", "export(model.eval(), torch.randn(2, 3, 224, 224)",
     "import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=4), "
     "torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())\n"),
    ("docs/KERNEL_CHECK.md", "check_kernel(reference, my_kernel, (x, w))",
     "import torch\ndef reference(x, w):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w\n"
     "my_kernel = reference\nx, w = torch.randn(16, 64), torch.rand(64) + 0.5\n"),
    ("docs/TOLERANCE.md", "truestar.assert_close(lambda x: torch.softmax(x, -1)",
     "import torch\nx = torch.randn(4, 32)\nmy_softmax = lambda x: torch.softmax(x, -1)\n"
     "my_ref = lambda a, b: a @ b\na, b = torch.randn(8, 16), torch.randn(16, 4)\nout = a @ b\n"),
    ("docs/NN.md", "loss(p, torch.randn(8, 16), torch.randint(0, 3, (8,)))",
     "import torch\nx, y = torch.randn(8, 16), torch.randint(0, 3, (8,))\n"),
    ("docs/AUTODIFF.md", 'loss, g = truestar.grad(rnn_loss, wrt="params")(params, x, y)',
     "import torch\nparams = {'wx': torch.randn(4, 5), 'wh': torch.randn(5, 5)}\n"
     "x, y = torch.randn(2, 3, 4), torch.randn(2, 5)\n"),
    ("docs/FRONTEND.md", "rnn(torch.randn(4, 9, 16), wx, wh)",
     "import torch\nwx, wh = torch.randn(16, 32) * 0.1, torch.randn(32, 32) * 0.1\n"),
]

# Names a reader cannot be expected to supply: a block that stops on one of these without importing
# it (here or in an earlier block of the same file) is missing an import.
IMPORT_NAMES = {"torch", "np", "numpy", "F", "nn", "copy", "math", "dataclass", "NamedTuple", "Dim",
                "truestar", "Tensor", "ops", "optim"}

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`\s]*)[^`]*$")
PY_LANGS = {"python", "py", "python3"}
CMD = re.compile(r"^(?:[A-Z_][A-Z0-9_]*=\S*\s+)*(?:truestar|python)\s")
CUDA_ERRORS = ("CUDA is not available", "Torch not compiled with CUDA", "no CUDA GPUs",
               "Found no NVIDIA driver", "needs CUDA", "requires CUDA", "needs a CUDA GPU",
               "No CUDA GPUs are available")
RESOLVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_docs_resolve.py")


def extract_blocks(text):
    """Fenced Python blocks of a markdown text: a list of (first code line number, code)."""
    blocks, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m.group("indent"), m.group("fence"), m.group("info").lower()
        body, j = [], i + 1
        while j < len(lines):
            close = re.match(r"^[ \t]*" + re.escape(fence[0]) + "{%d,}[ \t]*$" % len(fence), lines[j])
            if close:
                break
            line = lines[j]
            body.append(line[len(indent):] if line.startswith(indent) else line.lstrip())
            j += 1
        if lang in PY_LANGS:
            blocks.append((i + 2, "\n".join(body) + "\n"))
        i = j + 1
    return blocks


def docstring_commands(source):
    """The shell commands an example's module docstring lists, as written (comments removed)."""
    m = re.match(r'\s*(?:[rRuU]?)("""|\'\'\')(.*?)\1', source, re.S)
    if not m:
        return []
    doc, found = m.group(2), []
    # Inline commands in backticks (possibly wrapped over lines) that name an example file, so a
    # mention like `truestar kernel` or `truestar replay RUN` is not taken for a command.
    for span in re.finditer(r"`+([^`]+?)`+", doc):
        line = " ".join(span.group(1).split())
        if CMD.match(line) and "examples/" in line:
            found.append((span.start(), line))
    # Indented command lines with no backticks on them.
    pos = 0
    for raw in doc.splitlines(keepends=True):
        if "`" not in raw and CMD.match(raw.strip()):
            found.append((pos, raw.strip()))
        pos += len(raw)
    cmds = []
    for _, line in sorted(found):
        try:
            words = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if words:
            cmds.append(words)
    return cmds


def to_argv(words):
    """Shell words to (env, argv) for this interpreter: `truestar x` runs as `python -m truestar x`."""
    env, k = {}, 0
    while k < len(words) and re.match(r"^[A-Z_][A-Z0-9_]*=", words[k]):
        name, _, value = words[k].partition("=")
        if name != "PYTHONPATH":           # `PYTHONPATH=src` means this checkout's src, set already
            env[name] = value
        k += 1
    words = words[k:]
    if words[0] == "truestar":
        return env, [sys.executable, "-m", "truestar"] + words[1:]
    return env, [sys.executable] + words[1:]


def notebook_script(path):
    """The code cells of a notebook as one script; %%writefile cells become file writes."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        first, _, rest = src.partition("\n")
        if first.startswith("%%writefile"):
            name = first.split(None, 1)[1].strip()
            parts.append("open(%r, 'w', encoding='utf-8').write(%r)" % (name, rest))
            continue
        parts.append("\n".join("# " + ln if ln.lstrip().startswith(("%", "!")) else ln
                               for ln in src.splitlines()))
    return "\n\n".join(parts) + "\n"


def setup_for(item):
    for path, needle, code in SETUP:
        if path == item["rel"] and needle in item["text"]:
            return code
    return None


def skip_reason(rel, text):
    for path, needle, category, reason in SKIP:
        if path == rel and needle in text:
            return category, reason
    return None


def collect(only=None):
    """Every item to run: dicts with id, kind, rel, text and what to run."""
    items = []
    docs = [os.path.join(ROOT, n) for n in ("README_DRAFT.md", "README.md")]
    docs += sorted(os.path.join(ROOT, "docs", n) for n in os.listdir(os.path.join(ROOT, "docs"))
                   if n.endswith(".md"))
    for path in docs:
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as f:
            blocks = extract_blocks(f.read())
        for n, (line, code) in enumerate(blocks):
            items.append({"id": f"{rel}:{line}", "kind": "doc", "rel": rel, "text": code,
                          "code": code, "before": [c for _, c in blocks[:n]]})
    for name in sorted(os.listdir(EXAMPLES)):
        path = os.path.join(EXAMPLES, name)
        rel = f"examples/{name}"
        if name.endswith(".py"):
            with open(path, encoding="utf-8") as f:
                cmds = docstring_commands(f.read())
            text = "\n".join(" ".join(c) for c in cmds)
            items.append({"id": rel, "kind": "example", "rel": rel, "text": text, "cmds": cmds})
    nbdir = os.path.join(EXAMPLES, "notebooks")
    if os.path.isdir(nbdir):
        for name in sorted(os.listdir(nbdir)):
            if name.endswith(".ipynb"):
                code = notebook_script(os.path.join(nbdir, name))
                rel = f"examples/notebooks/{name}"
                items.append({"id": rel, "kind": "notebook", "rel": rel, "text": code, "code": code,
                              "cwd": nbdir})
    for it in items:
        if it["kind"] == "example" and not it["cmds"]:
            users = [o["id"] for o in items if o["kind"] == "example" and it["rel"] in o["text"]]
            if users:
                it["covered_by"] = users[0]
    if only:
        items = [it for it in items if only in it["id"]]
    return items


def child_env(extra=None):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=SRC, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env.update(extra or {})
    return env


def run_proc(argv, cwd, timeout, env=None):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(argv, cwd=cwd, env=child_env(env), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return None, out, err, time.perf_counter() - t0


def run_code(code, work, timeout):
    path = os.path.join(work, "snippet.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return run_proc([sys.executable, path], work, timeout)


def has_traceback(text):
    return "Traceback (most recent call last)" in text


def resolve(item, work):
    """Static check of a block (with the blocks before it as context): {"syntax"} or {"problems"}."""
    tried = [("\n\n".join(item["before"] + [item["code"]]), None)] if item["before"] else []
    tried.append((item["code"], 1))
    for code, first in tried:
        if first is None:
            first = code.count("\n", 0, len(code) - len(item["code"])) + 1
        path = os.path.join(work, "resolve_me.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err, _ = run_proc([sys.executable, RESOLVER, path, str(first)], work, 120)
        try:
            doc = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"problems": [{"line": 0, "message": "resolver failed: " + (out + err)[-300:]}]}
        if "syntax" not in doc or first == 1:
            return doc
    return doc


def grade_sketch(res, item, err, work):
    """A failing doc block: a sketch with the reader's own objects, a GPU block, or a real failure.

    The block counts as a sketch (status fragment) or a GPU block (status gpu) only when every
    truestar import, attribute and keyword argument in it resolves; otherwise it fails with the
    names that do not exist.
    """
    doc = resolve(item, work)
    res["resolve"] = doc
    if doc.get("problems"):
        res["reason"] = "; ".join(f"line {p['line']}: {p['message']}" for p in doc["problems"])
        return
    if "syntax" in doc:
        res["reason"] = "not valid Python: " + doc["syntax"]
        return
    last = err.strip().splitlines()[-1] if err.strip() else ""
    m = re.match(r"NameError: name '(\w+)' is not defined", last)
    if m:
        name = m.group(1)
        earlier = "\n".join(item["before"])
        if re.search(r"^\s*(?:import|from)\s.*\b%s\b" % re.escape(name), earlier, re.M):
            why = f"continues the blocks before it (they import {name!r})"
        elif name in IMPORT_NAMES:
            res["reason"] = f"missing import: {name!r} is used but not imported here or earlier"
            return
        else:
            why = f"uses the reader's {name!r}"
        res.update(status="fragment", missing=name,
                   reason=why + "; truestar names and keywords resolve")
    elif re.match(r"FileNotFoundError: .*'([^']+)'", last) and os.path.basename(
            re.match(r"FileNotFoundError: .*'([^']+)'", last).group(1)) in item["code"]:
        name = os.path.basename(re.match(r"FileNotFoundError: .*'([^']+)'", last).group(1))
        res.update(status="fragment", missing=name,
                   reason=f"reads the reader's file {name!r}; truestar names and keywords resolve")
    elif any(h in last for h in CUDA_ERRORS):
        res.update(status="gpu", reason="needs CUDA; truestar names and keywords resolve")


def run_item(item, timeout=120.0):
    """Run one item; returns a result dict with status pass, fail, timeout or skip."""
    res = {"id": item["id"], "kind": item["kind"]}
    if item.get("covered_by"):
        res.update(status="skip", category="covered", reason="runs as part of " + item["covered_by"])
        return res
    if item["kind"] == "example":
        cmds, skipped = [], []
        for words in item["cmds"]:
            why = skip_reason(item["rel"], " ".join(words))
            (skipped if why else cmds).append((words, why))
        if skipped:
            res["skipped"] = [{"cmd": " ".join(w), "category": y[0], "reason": y[1]} for w, y in skipped]
        if item["cmds"] and not cmds:
            res.update(status="skip", category=skipped[0][1][0], reason=skipped[0][1][1])
            return res
        item = dict(item, cmds=[w for w, _ in cmds])
    else:
        skip = skip_reason(item["rel"], item["text"])
        if skip:
            res.update(status="skip", category=skip[0], reason=skip[1])
            return res
    work = tempfile.mkdtemp(prefix="ts_docs_")
    try:
        shutil.copytree(EXAMPLES, os.path.join(work, "examples"))
        if item["kind"] == "example":
            if not item["cmds"]:
                res.update(status="fail", reason="docstring lists no command to run")
                return res
            total, steps = 0.0, []
            for words in item["cmds"]:
                env, argv = to_argv(words)
                rc, out, err, took = run_proc(argv, work, timeout, env)
                total += took
                steps.append({"cmd": " ".join(words), "rc": rc, "seconds": round(took, 1),
                              "tail": (out + err)[-1500:]})
                if rc is None or rc not in (0, 1) or has_traceback(err):
                    break
            res["steps"] = steps
            last = steps[-1]
            bad = last["rc"] is None or last["rc"] not in (0, 1) or has_traceback(last["tail"])
            status = "timeout" if last["rc"] is None else ("fail" if bad else "pass")
            if last["rc"] not in (0, None) and not has_traceback(last["tail"]) and any(
                    h in last["tail"] for h in CUDA_ERRORS):
                status = "gpu"              # stopped up front, saying it needs CUDA
            res.update(status=status, seconds=round(total, 1), rc=last["rc"], tail=last["tail"])
            return res
        cwd = item.get("cwd", work)
        if item["kind"] == "notebook":
            path = os.path.join(work, "snippet.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(item["code"])
            rc, out, err, took = run_proc([sys.executable, path], cwd, timeout)
        else:
            rc, out, err, took = run_code(item["code"], work, timeout)
            if rc not in (0, None) and "NameError" in err and item["before"]:
                rc2, out2, err2, took2 = run_code("\n\n".join(item["before"] + [item["code"]]), work,
                                                  timeout)
                res["with_earlier_blocks"] = True
                if rc2 == 0:
                    rc, out, err, took = rc2, out2, err2, took + took2
                else:
                    res["earlier_blocks_tail"] = (out2 + err2)[-800:]
        status = "timeout" if rc is None else ("pass" if rc == 0 else "fail")
        res.update(status=status, rc=rc, seconds=round(took, 1), tail=(out + err)[-1500:])
        if status == "fail" and item["kind"] == "doc":
            grade_sketch(res, item, err, work)
            setup = setup_for(item)
            if res["status"] == "fragment" and setup:
                rc, out, err, took = run_code(setup + "\n" + item["code"], work, timeout)
                res.update(rc=rc, seconds=round(res["seconds"] + took, 1), tail=(out + err)[-1500:])
                if rc == 0:
                    res.update(status="setup",
                               reason=f"runs with a small CPU setup for {res.get('missing')!r}")
                else:
                    last = (err.strip().splitlines() or ["?"])[-1][:300]
                    if "BackendUnavailable" in last and "cuda" in last:
                        # the block asks for backends=[..., "cuda"]; refusing that here is right
                        res.update(status="gpu", reason="asks for the cuda backend; with a small CPU "
                                                        "setup it stops only on that: " + last)
                    else:
                        res.update(status="timeout" if rc is None else "fail",
                                   reason="fails with a small CPU setup: " + last)
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="run only items whose id contains this text")
    ap.add_argument("--json", help="write every result to this file")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--list", action="store_true", help="list the items and exit")
    args = ap.parse_args(argv)
    items = collect(args.only)
    if args.list:
        for it in items:
            print(it["id"], it["kind"])
        print(f"{len(items)} items")
        return 0
    results, counts = [], {}
    for it in items:
        r = run_item(it, args.timeout)
        results.append(r)
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        extra = r.get("reason") or r.get("hint") or ""
        print(f"{r['status']:8} {r.get('seconds', 0):6.1f}s {r['id']}  {extra}".rstrip(), flush=True)
        if r["status"] in ("fail", "timeout"):
            print("    " + r.get("tail", "").strip().replace("\n", "\n    ")[-600:], flush=True)
    print("counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"counts": counts, "results": results}, f, indent=1)
    return 0 if not (counts.get("fail") or counts.get("timeout")) else 1


if __name__ == "__main__":
    sys.exit(main())
