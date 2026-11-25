"""Nested inputs and outputs (pytrees) for the native front end and for exported artifacts.

A structure is a tree of tuples, lists, dicts, NamedTuples and dataclasses whose leaves are tensors
(or, in an output, ``None`` and plain numbers). Its spec is plain JSON data, so an artifact stores
it next to the program and a fresh process rebuilds the same structure without importing user
code. Dataclasses need no registration: their fields are read with ``dataclasses.fields``.

Every mismatch names the exact leaf path, written the way the user would index it::

    inputs['params']['w']: missing key 'w' (given keys: 'bias', 'weight')
    batch.x: expected a tensor f32[B, 16], got a list
"""

from __future__ import annotations

import collections
import dataclasses
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Spec = Dict[str, Any]
LEAF: Spec = {"kind": "leaf"}


class StructureError(TypeError):
    """A value does not have the structure the signature declares. ``path`` names the leaf."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(f"{path}: {message}" if path else message)


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(type(x), "_fields")


def _is_dataclass_instance(x) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def key_path(path: str, key: Any) -> str:
    return f"{path}[{key}]" if isinstance(key, int) else f"{path}[{key!r}]"


def attr_path(path: str, name: str) -> str:
    return f"{path}.{name}"


def _type_name(x) -> str:
    return type(x).__name__


def flatten(obj: Any, is_leaf: Callable[[Any], bool], path: str = "", keep_class: bool = False) -> Tuple[List[Any], Spec]:
    """(leaves, spec). ``is_leaf`` says what counts as a leaf (a tensor, a traced value).
    ``keep_class`` also keeps the class object of NamedTuple and dataclass nodes in the spec (in
    memory only, so unflatten rebuilds that very class; never written to a file)."""
    leaves, spec = _flatten(obj, is_leaf, path)
    if not keep_class:
        _drop_classes(spec)
    return leaves, spec


def _drop_classes(spec: Spec) -> None:
    spec.pop("cls", None)
    for c in spec.get("children", []):
        _drop_classes(c)


def _flatten(obj: Any, is_leaf: Callable[[Any], bool], path: str = "") -> Tuple[List[Any], Spec]:
    if is_leaf(obj):
        return [obj], dict(LEAF)
    if obj is None:
        return [], {"kind": "none"}
    if isinstance(obj, (bool, int, float, str)):
        return [], {"kind": "const", "value": obj}
    if _is_namedtuple(obj):
        leaves, kids = [], []
        for f in obj._fields:
            lv, sp = _flatten(getattr(obj, f), is_leaf, attr_path(path, f))
            leaves += lv
            kids.append(sp)
        return leaves, {"kind": "namedtuple", "name": _type_name(obj), "module": type(obj).__module__,
                        "fields": list(obj._fields), "children": kids, "cls": type(obj)}
    if isinstance(obj, (tuple, list)):
        leaves, kids = [], []
        for i, x in enumerate(obj):
            lv, sp = _flatten(x, is_leaf, key_path(path, i))
            leaves += lv
            kids.append(sp)
        return leaves, {"kind": "tuple" if isinstance(obj, tuple) else "list", "children": kids}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        for k in keys:
            if not isinstance(k, (str, int)):
                raise StructureError(path, f"dict keys must be str or int, got {k!r}")
        leaves, kids = [], []
        for k in keys:
            lv, sp = _flatten(obj[k], is_leaf, key_path(path, k))
            leaves += lv
            kids.append(sp)
        return leaves, {"kind": "dict", "keys": keys, "children": kids}
    if _is_dataclass_instance(obj):
        names = [f.name for f in dataclasses.fields(obj)]
        leaves, kids = [], []
        for f in names:
            lv, sp = _flatten(getattr(obj, f), is_leaf, attr_path(path, f))
            leaves += lv
            kids.append(sp)
        return leaves, {"kind": "dataclass", "name": _type_name(obj), "module": type(obj).__module__,
                        "fields": names, "children": kids, "cls": type(obj)}
    raise StructureError(path, f"a {_type_name(obj)} cannot be part of an input or output; use tensors in "
                               "tuples, lists, dicts, NamedTuples or dataclasses")


def none_leaves(spec: Spec, positions) -> Spec:
    """The spec with the leaves at these flat positions turned into None (torch.export counts a None
    output as a leaf; Truestar stores it as structure, since it carries no tensor)."""
    positions = set(positions)
    count = [0]

    def walk(sp):
        if sp["kind"] == "leaf":
            i = count[0]
            count[0] += 1
            return {"kind": "none"} if i in positions else sp
        if "children" in sp:
            return {**sp, "children": [walk(c) for c in sp["children"]]}
        return sp
    return walk(spec)


def num_leaves(spec: Spec) -> int:
    if spec["kind"] == "leaf":
        return 1
    return sum(num_leaves(c) for c in spec.get("children", []))


def leaf_paths(spec: Spec, path: str = "") -> List[str]:
    """The path of every leaf, in flattening order."""
    k = spec["kind"]
    if k == "leaf":
        return [path]
    out: List[str] = []
    if k in ("tuple", "list"):
        for i, c in enumerate(spec["children"]):
            out += leaf_paths(c, key_path(path, i))
    elif k == "dict":
        for key, c in zip(spec["keys"], spec["children"]):
            out += leaf_paths(c, key_path(path, key))
    elif k in ("namedtuple", "dataclass"):
        for f, c in zip(spec["fields"], spec["children"]):
            out += leaf_paths(c, attr_path(path, f))
    return out


def describe(spec: Spec) -> str:
    """A short readable form: ``{'w': T, 'b': T}``, ``(T, [T, T])``, ``Out(logits=T, loss=None)``."""
    k = spec["kind"]
    if k == "leaf":
        return "T"
    if k == "none":
        return "None"
    if k == "const":
        return repr(spec["value"])
    kids = [describe(c) for c in spec.get("children", [])]
    if k == "tuple":
        return "(" + ", ".join(kids) + ("," if len(kids) == 1 else "") + ")"
    if k == "list":
        return "[" + ", ".join(kids) + "]"
    if k == "dict":
        return "{" + ", ".join(f"{key!r}: {v}" for key, v in zip(spec["keys"], kids)) + "}"
    return f"{spec['name']}(" + ", ".join(f"{f}={v}" for f, v in zip(spec["fields"], kids)) + ")"


def match(spec: Spec, obj: Any, is_leaf: Callable[[Any], bool], path: str = "",
          check_leaf: Optional[Callable[[str, Any, int], None]] = None, _count: Optional[List[int]] = None) -> List[Any]:
    """Leaves of ``obj`` in the order of ``spec``, checking the structure as it goes. Raises
    StructureError naming the first leaf path that differs. ``check_leaf(path, leaf, i)`` may
    check each leaf (dtype, shape) and raise itself."""
    count = _count if _count is not None else [0]
    k = spec["kind"]
    if k == "leaf":
        if not is_leaf(obj):
            raise StructureError(path, f"expected a tensor, got {'None' if obj is None else 'a ' + _type_name(obj)}")
        if check_leaf is not None:
            check_leaf(path, obj, count[0])
        count[0] += 1
        return [obj]
    if k == "none":
        if obj is not None:
            raise StructureError(path, f"expected None, got a {_type_name(obj)}")
        return []
    if k == "const":
        if obj != spec["value"]:
            raise StructureError(path, f"expected the constant {spec['value']!r}, got {obj!r}")
        return []
    out: List[Any] = []
    if k in ("tuple", "list"):
        if not isinstance(obj, (tuple, list)):
            raise StructureError(path, f"expected a {k} of {len(spec['children'])}, got a {_type_name(obj)}")
        if len(obj) != len(spec["children"]):
            raise StructureError(path, f"expected {len(spec['children'])} items, got {len(obj)}")
        for i, (c, x) in enumerate(zip(spec["children"], obj)):
            out += match(c, x, is_leaf, key_path(path, i), check_leaf, count)
        return out
    if k == "dict":
        if not isinstance(obj, dict):
            raise StructureError(path, f"expected a dict with keys {spec['keys']}, got a {_type_name(obj)}")
        want, got = list(spec["keys"]), list(obj.keys())
        missing = [x for x in want if x not in obj]
        extra = [x for x in got if x not in want]
        if missing or extra:
            parts = []
            if missing:
                parts.append("missing key" + ("s " if len(missing) > 1 else " ") + ", ".join(map(repr, missing)))
            if extra:
                parts.append("unexpected key" + ("s " if len(extra) > 1 else " ") + ", ".join(map(repr, extra)))
            where = key_path(path, missing[0]) if missing else key_path(path, extra[0])
            raise StructureError(where, "; ".join(parts) + f" (expected keys: {', '.join(map(repr, want))})")
        for key, c in zip(want, spec["children"]):
            out += match(c, obj[key], is_leaf, key_path(path, key), check_leaf, count)
        return out
    if k in ("namedtuple", "dataclass"):
        fields = spec["fields"]
        if isinstance(obj, dict):
            raise StructureError(path, f"expected a {spec['name']}, got a dict")
        for f in fields:
            if not hasattr(obj, f):
                raise StructureError(attr_path(path, f), f"missing field {f!r} (a {spec['name']} has fields "
                                                         f"{', '.join(fields)}; got a {_type_name(obj)})")
        for f, c in zip(fields, spec["children"]):
            out += match(c, getattr(obj, f), is_leaf, attr_path(path, f), check_leaf, count)
        return out
    raise StructureError(path, f"unknown structure kind {k!r}")


def _rebuild_class(spec: Spec):
    """The user's class if its module is already imported (never imported here: loading an
    artifact must not run code), else a stand in with the same name and fields."""
    if spec.get("cls") is not None:
        return spec["cls"]
    mod = sys.modules.get(spec.get("module") or "")
    cls = getattr(mod, spec["name"], None) if mod is not None else None
    if cls is not None:
        if spec["kind"] == "namedtuple" and getattr(cls, "_fields", None) == tuple(spec["fields"]):
            return cls
        if spec["kind"] == "dataclass" and dataclasses.is_dataclass(cls) and \
                [f.name for f in dataclasses.fields(cls)] == list(spec["fields"]):
            return cls
    if spec["kind"] == "namedtuple":
        return collections.namedtuple(spec["name"], spec["fields"])
    return dataclasses.make_dataclass(spec["name"], spec["fields"])


def unflatten(spec: Spec, leaves: Sequence[Any]) -> Any:
    it = iter(leaves)
    out = _unflatten(spec, it)
    rest = list(it)
    if rest:
        raise StructureError("", f"{len(rest)} leaves left over after rebuilding {describe(spec)}")
    return out


def _unflatten(spec: Spec, it) -> Any:
    k = spec["kind"]
    if k == "leaf":
        try:
            return next(it)
        except StopIteration:
            raise StructureError("", f"too few leaves to rebuild the structure") from None
    if k == "none":
        return None
    if k == "const":
        return spec["value"]
    kids = [_unflatten(c, it) for c in spec.get("children", [])]
    if k == "tuple":
        return tuple(kids)
    if k == "list":
        return kids
    if k == "dict":
        return dict(zip(spec["keys"], kids))
    if k in ("namedtuple", "dataclass"):
        cls = _rebuild_class(spec)
        if k == "namedtuple":
            return cls(*kids)
        obj = object.__new__(cls)
        for f, v in zip(spec["fields"], kids):
            object.__setattr__(obj, f, v)
        return obj
    raise StructureError("", f"unknown structure kind {k!r}")


def from_torch_spec(ts) -> Spec:
    """A torch.utils._pytree.TreeSpec as a Truestar spec (what torch.export records in its call spec)."""
    if ts.is_leaf():
        return dict(LEAF)
    t = ts.type
    children = getattr(ts, "children", None)       # torch 2.11 deprecated children_specs
    kids = [from_torch_spec(c) for c in (children() if callable(children) else ts.children_specs)]
    if t is type(None):
        return {"kind": "none"}
    if t is tuple:
        return {"kind": "tuple", "children": kids}
    if t is list:
        return {"kind": "list", "children": kids}
    if t in (dict, collections.OrderedDict, collections.defaultdict):
        keys = list(ts.context) if t is not collections.defaultdict else list(ts.context[1])
        return {"kind": "dict", "keys": keys, "children": kids}
    if isinstance(ts.context, type) and issubclass(ts.context, tuple) and hasattr(ts.context, "_fields"):
        t = ts.context  # torch files every NamedTuple under one node type, the class is the context
    if isinstance(t, type) and issubclass(t, tuple) and hasattr(t, "_fields"):
        return {"kind": "namedtuple", "name": t.__name__, "module": t.__module__, "fields": list(t._fields),
                "children": kids}
    if dataclasses.is_dataclass(t):
        return {"kind": "dataclass", "name": t.__name__, "module": t.__module__,
                "fields": [f.name for f in dataclasses.fields(t)], "children": kids}
    ctx = ts.context
    if isinstance(ctx, (list, tuple)) and all(isinstance(x, str) for x in ctx) and len(ctx) == len(kids):
        # a registered mapping such as a Hugging Face ModelOutput: rebuilt as a dict with the same keys
        return {"kind": "dict", "keys": list(ctx), "children": kids, "from": f"{t.__module__}.{t.__qualname__}"}
    raise StructureError("", f"torch pytree node {t.__qualname__} has no Truestar equivalent; return tensors in "
                             "tuples, lists, dicts, NamedTuples or dataclasses")
