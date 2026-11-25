"""Child process for the cross framework port check: runs a JAX/Flax NNX or Keras 3 candidate.

Keras fixes its backend when it is first imported, and JAX's float64 switch must be set before any
array exists, so every candidate framework (and every Keras backend) runs in its own process. The
parent sends one JSON request per line on stdin and reads one JSON reply per line on stdout;
arrays travel in safetensors files written and read by Truestar (truestar.interop), and inside this
process framework arrays become torch tensors through DLPack (truestar.interop.from_dlpack, zero
copy and checked) before they are saved.

Requests:
  {"op": "load", "framework": "jax"|"keras", "file": cand.py, "weights": w.safetensors,
   "inputs": in.safetensors, "out": dir}
      builds the candidate with get_model(weights), runs it on the inputs while recording every
      module call, exports its parameters. Reply: modules, calls, files.
  {"op": "replay", "jobs": [{"module": path, "args": [key or null], "file": args.safetensors}], "out": f}
      runs each module alone on the given arguments. Reply: output keys.
  {"op": "quit"}
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple


def _reply(obj: Dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


class Framework:
    name = ""

    def to_native(self, t):
        raise NotImplementedError

    def to_torch(self, a):
        from truestar.interop import from_dlpack
        import numpy as np
        import torch

        if isinstance(a, torch.Tensor):
            return a.detach().cpu()
        try:
            return from_dlpack(a).cpu().clone()
        except Exception:
            return torch.from_numpy(np.array(a))

    def is_array(self, a) -> bool:
        raise NotImplementedError

    def modules(self, model) -> List[Tuple[str, Any]]:
        raise NotImplementedError

    def own_params(self, path: str, module, children) -> Dict[str, Any]:
        raise NotImplementedError

    def patch(self, mods, record):
        raise NotImplementedError

    def call(self, module, args):
        return module(*args)


class JaxFramework(Framework):
    name = "jax"

    def __init__(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platforms", "cpu")
        import jax.numpy as jnp
        from flax import nnx

        self.jax, self.jnp, self.nnx = jax, jnp, nnx

    def version(self) -> str:
        import flax

        return f"jax {self.jax.__version__}, flax {flax.__version__}"

    def to_native(self, t):
        import jax.dlpack

        return jax.dlpack.from_dlpack(t.contiguous())

    def is_array(self, a) -> bool:
        return isinstance(a, self.jax.Array)

    def modules(self, model):
        out = []
        for path, m in model.iter_modules():
            out.append((".".join(str(p) for p in path), m))
        return out

    def own_params(self, path, module, children):
        out = {}
        for k, v in vars(module).items():
            if isinstance(v, self.nnx.Param):
                out[(path + "." if path else "") + k] = v.value
        return out

    def patch(self, mods, record):
        ids = {id(m): p for p, m in mods}
        classes = {type(m) for _, m in mods}
        saved = []
        for cls in classes:
            orig = cls.__dict__.get("__call__")
            if orig is None:
                continue

            def wrapper(self, *a, _orig=orig, **k):
                p = ids.get(id(self))
                if p is None:
                    return _orig(self, *a, **k)
                tok = record.enter(p, type(self).__name__, a)
                out = _orig(self, *a, **k)
                record.exit(tok, out)
                return out
            saved.append((cls, orig))
            cls.__call__ = wrapper
        return lambda: [setattr(c, "__call__", o) for c, o in saved]


class KerasFramework(Framework):
    def __init__(self):
        if os.environ.get("KERAS_BACKEND") == "jax":
            import jax

            jax.config.update("jax_enable_x64", True)  # before keras creates any array
        import keras

        self.keras = keras
        self.name = "keras:" + keras.backend.backend()
        keras.config.set_floatx("float64")

    def version(self) -> str:
        return f"keras {self.keras.__version__} ({self.keras.backend.backend()} backend)"

    def to_native(self, t):
        b = self.keras.backend.backend()
        if b == "torch":
            return t
        if b == "jax":
            import jax.dlpack

            return jax.dlpack.from_dlpack(t.contiguous())
        return self.keras.ops.convert_to_tensor(t.numpy())

    def is_array(self, a) -> bool:
        import torch

        if self.keras.backend.backend() == "torch":
            return isinstance(a, torch.Tensor)
        return self.keras.ops.is_tensor(a) and not isinstance(a, (int, float))

    def modules(self, model):
        out = []
        seen = set()

        def walk(path, layer):
            if id(layer) in seen:
                return
            seen.add(id(layer))
            out.append((path, layer))
            items = list(vars(layer).items())
            mods = vars(layer).get("_modules")  # torch backend: sublayers are torch submodules
            if isinstance(mods, dict):
                items += [(k, v) for k, v in mods.items() if k not in vars(layer)]
            for k, v in items:
                if k.startswith("_"):
                    continue
                if isinstance(v, self.keras.layers.Layer):
                    walk((path + "." if path else "") + k, v)
                elif isinstance(v, (list, tuple)):
                    for i, x in enumerate(v):
                        if isinstance(x, self.keras.layers.Layer):
                            walk((path + "." if path else "") + f"{k}.{i}", x)
        walk("", model)
        return out

    def own_params(self, path, module, children):
        inner = {id(v) for c in children for v in c.weights}
        out = {}
        for v in module.weights:
            if id(v) in inner:
                continue
            out[(path + "." if path else "") + v.name] = v.value
        return out

    def patch(self, mods, record):
        saved = []
        for p, layer in mods:
            orig = layer.call

            def wrapper(*a, _orig=orig, _p=p, _l=layer, **k):
                tok = record.enter(_p, type(_l).__name__, a)
                out = _orig(*a, **k)
                record.exit(tok, out)
                return out
            layer.call = wrapper
            saved.append(layer)
        return lambda: [l.__dict__.pop("call", None) for l in saved]


class Record:
    def __init__(self, fw: Framework):
        self.fw = fw
        self.calls: List[Dict[str, Any]] = []
        self.tensors: Dict[str, Any] = {}
        self.stack: List[int] = []
        self.counts: Dict[str, int] = {}

    def _leaves(self, obj) -> List[Any]:
        if isinstance(obj, (list, tuple)):
            return [x for o in obj for x in self._leaves(o)]
        if isinstance(obj, dict):
            return [x for k in sorted(obj, key=str) for x in self._leaves(obj[k])]
        return [obj] if self.fw.is_array(obj) else []

    def enter(self, path, type_name, args):
        idx = self.counts.get(path, 0)
        self.counts[path] = idx + 1
        n = len(self.calls)
        arg_keys = []
        for j, a in enumerate(args):
            if self.fw.is_array(a):
                key = f"c{n}.in{j}"
                self.tensors[key] = self.fw.to_torch(a)
                arg_keys.append(key)
            elif a is None or isinstance(a, (bool, int, float, str)):
                arg_keys.append({"value": a})
            else:
                arg_keys.append({"opaque": type(a).__name__})
        self.calls.append({"module": path, "type": type_name, "index": idx, "args": arg_keys,
                           "parent": self.stack[-1] if self.stack else None, "outputs": []})
        self.stack.append(n)
        return n

    def exit(self, n, out):
        self.stack.pop()
        keys = []
        for j, a in enumerate(self._leaves(out)):
            key = f"c{n}.out{j}"
            self.tensors[key] = self.fw.to_torch(a)
            keys.append(key)
        self.calls[n]["outputs"] = keys


def _import(path):
    spec = importlib.util.spec_from_file_location("truestar_xport_candidate", path)
    mod = importlib.util.module_from_spec(spec)
    d = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model"):
        raise ValueError(f"{path} must define get_model(weights_path)")
    return mod


def main() -> int:
    sys.stdout = sys.stderr  # candidate prints never corrupt the protocol
    fw = None
    model = None
    mods: List[Tuple[str, Any]] = []
    from truestar.interop import load_file, save_file
    import torch

    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req["op"] == "quit":
                _reply({"ok": True})
                return 0
            if req["op"] == "load":
                fw = JaxFramework() if req["framework"] == "jax" else KerasFramework()
                cand = _import(req["file"])
                model = cand.get_model(req["weights"])
                mods = fw.modules(model)
                by_path = dict(mods)
                children = {p: [m for q, m in mods if q != p and (p == "" or q.startswith(p + "."))] for p, _ in mods}
                params: Dict[str, Any] = {}
                owned: Dict[str, List[str]] = {}
                for p, m in mods:
                    own = fw.own_params(p, m, [c for c in children[p]])
                    owned[p] = list(own)
                    for k, v in own.items():
                        params[k] = fw.to_torch(v).to(torch.float64)
                inputs = load_file(req["inputs"], mmap=False)
                args = [fw.to_native(inputs[f"in{i}"].clone()) for i in range(len(inputs))]
                rec = Record(fw)
                undo = fw.patch(mods, rec)
                try:
                    out = fw.call(model, args)
                finally:
                    undo()
                final = [fw.to_torch(a) for a in rec._leaves(out)]
                os.makedirs(req["out"], exist_ok=True)
                pf = os.path.join(req["out"], "params.safetensors")
                rf = os.path.join(req["out"], "records.safetensors")
                save_file({k: v.contiguous() for k, v in params.items()}, pf)
                tens = dict(rec.tensors)
                for i, t in enumerate(final):
                    tens[f"final{i}"] = t
                save_file({k: v.contiguous() for k, v in tens.items()}, rf)
                _reply({"ok": True, "version": fw.version(), "framework": fw.name,
                        "modules": [{"path": p, "type": type(m).__name__, "params": owned[p]} for p, m in mods],
                        "calls": rec.calls, "params": pf, "records": rf, "final": [f"final{i}" for i in range(len(final))]})
                del by_path
            elif req["op"] == "replay":
                by_path = dict(mods)
                outs: Dict[str, Any] = {}
                for j, job in enumerate(req["jobs"]):
                    arrs = load_file(job["file"], mmap=False)
                    args = [fw.to_native(arrs[k].clone()) if isinstance(k, str) else k.get("value") for k in job["args"]]
                    res = fw.call(by_path[job["module"]], args)
                    for i, a in enumerate(Record(fw)._leaves(res)):
                        outs[f"j{j}.out{i}"] = fw.to_torch(a).contiguous()
                save_file(outs, req["out"])
                _reply({"ok": True, "outputs": sorted(outs)})
            else:
                _reply({"ok": False, "error": f"unknown op {req['op']}"})
        except Exception as exc:
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
