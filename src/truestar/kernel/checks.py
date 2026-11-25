"""Compiled in runtime checks: the classes, the flag layout, and the static analysis that keeps
deliberate infinities (attention masks) from raising false alarms.

A kernel compiled with checks writes one flag word per check class into a small int32 buffer the
caller passes (``_chk``). Each word is written only when a block finds a bad value, so a clean run
costs one block reduction per output and per checked division, and no global store at all. The
caller reads the buffer once per call; finding which op, which line and which element is the
caller's job (``truestar.compiler.checks``), done only after a flag fired.

Classes (``CLASSES``, the slot of each word in that order):

* ``nan``: a stored output holds NaN.
* ``inf``: a stored output holds +Inf or -Inf in the register precision (float32 on the GPU).
* ``overflow``: a float16 or bfloat16 output rounds to Inf on store while its float32 value was
  finite (the value did not fit the storage precision).
* ``div``: a division whose denominator is exactly zero, at any point of the fused expression.
  Denominators that cannot be zero are not checked: constants, sizes, and sums of exponentials
  (the softmax normalizer).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Set

from .ir import Binary, Const, Dim, Expr, Load, Placeholder, Reduce, Unary, Where

CLASSES = ("nan", "inf", "overflow", "div")
SLOT = {c: i for i, c in enumerate(CLASSES)}
WORDS = len(CLASSES)
DEFAULT = ("nan", "div")          # the default set: NaN origins and zero denominators
ALIASES = {"all": CLASSES, "float": DEFAULT, "default": DEFAULT, "true": DEFAULT, "1": DEFAULT,
           "nan_inf": ("nan", "inf"), "finite": ("nan", "inf")}


def parse(checks) -> FrozenSet[str]:
    """``None``, ``False``, ``""``, ``"off"``: no checks. ``True``: nan and div. ``"all"``: every
    class. A comma separated string or an iterable names classes (``"nan,inf,div,overflow"``).
    ``bounds`` is accepted and ignored: index bounds are always checked."""
    if checks is None or checks is False:
        return frozenset()
    if checks is True:
        return frozenset(DEFAULT)
    if isinstance(checks, str):
        text = checks.strip().lower()
        if text in ("", "0", "off", "none", "false", "no"):
            return frozenset()
        parts = [p.strip() for p in text.replace("+", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in checks]
    out: Set[str] = set()
    for p in parts:
        if p in ALIASES:
            out.update(ALIASES[p])
        elif p in CLASSES:
            out.add(p)
        elif p in ("bounds", "index"):
            continue
        else:
            raise ValueError(f"unknown check class {p!r}; use any of {', '.join(CLASSES)}, 'bounds' "
                             f"(always on), 'all' or True (nan and div)")
    return frozenset(out)


def text(classes: Iterable[str]) -> str:
    """Canonical text of a set of classes (part of the kernel cache key)."""
    return ",".join(c for c in CLASSES if c in set(classes))


# ----------------------------------------------------------------------------- denominators


def may_be_zero(den: Expr) -> bool:
    """False for denominators that cannot be exactly zero by construction: nonzero constants,
    sizes, and a sum of exponentials (softmax), which is at least 1 after the max is subtracted."""
    if isinstance(den, Const):
        return den.value == 0 or den.value != den.value
    if isinstance(den, Dim):
        return False
    if isinstance(den, Reduce) and den.op == "sum" and isinstance(den.body, Unary) and den.body.op == "exp":
        return False
    if isinstance(den, Placeholder):
        return False
    return True


# ----------------------------------------------------------------------------- infinities

POS, NEG = "+", "-"
_BOTH = frozenset((POS, NEG))
_NONE: FrozenSet[str] = frozenset()


def may_be_inf(e: Expr, tainted: Dict[str, FrozenSet[str]], memo: Optional[dict] = None) -> FrozenSet[str]:
    """Signs of infinity that ``e`` can hold *on purpose*: from an infinite constant or from a
    buffer in ``tainted`` (name -> signs), carried through the ops that keep it. An infinity that
    arises from finite values (an overflow, log(0), x / 0) is not counted: that is what the check
    is for. ``exp(-inf)`` is 0, so the masked softmax of an attention block is finite."""
    if memo is None:
        memo = {}
    hit = memo.get(e)
    if hit is not None:
        return hit
    r = _NONE
    if isinstance(e, Const):
        if isinstance(e.value, float) and e.value in (float("inf"), float("-inf")):
            r = frozenset((POS if e.value > 0 else NEG,))
    elif isinstance(e, Load):
        r = tainted.get(e.buffer.name, _NONE)
    elif isinstance(e, Unary):
        x = may_be_inf(e.x, tainted, memo)
        if e.op == "neg":
            r = frozenset({POS: NEG, NEG: POS}[s] for s in x)
        elif e.op in ("exp", "expm1"):
            r = frozenset((POS,)) if POS in x else _NONE
        elif e.op in ("log", "sqrt"):
            r = frozenset((POS,)) if POS in x else _NONE
        elif e.op == "abs":
            r = frozenset((POS,)) if x else _NONE
        elif e.op in ("sigmoid", "tanh", "erf", "not", "cos", "sin", "rsqrt"):
            r = _NONE
        else:
            r = x
    elif isinstance(e, Binary):
        a, b = may_be_inf(e.a, tainted, memo), may_be_inf(e.b, tainted, memo)
        if e.op in ("lt", "le", "gt", "ge", "eq", "ne", "and", "or"):
            r = _NONE
        elif e.op == "add":
            r = a | b
        elif e.op == "sub" and isinstance(e.b, Reduce) and e.b.op == "max":
            # the softmax shift s - max_r s is at most 0: a masked -inf stays -inf (exp gives 0);
            # a row with every entry masked gives NaN, which the nan check reports
            r = a & {NEG}
        elif e.op == "sub":
            r = a | frozenset({POS: NEG, NEG: POS}[s] for s in b)
        elif e.op == "mul":
            r = _BOTH if (a or b) else _NONE
        elif e.op == "div":
            r = _BOTH if a else _NONE
        elif e.op == "maximum":
            r = (a & {POS}) | (b & {POS}) | (a & b & {NEG})
        elif e.op == "minimum":
            r = (a & {NEG}) | (b & {NEG}) | (a & b & {POS})
        else:
            r = a | b
    elif isinstance(e, Where):
        r = may_be_inf(e.a, tainted, memo) | may_be_inf(e.b, tainted, memo)
    elif isinstance(e, Reduce):
        r = may_be_inf(e.body, tainted, memo)
    else:
        r = _NONE
        for c in e.children():
            r = r | may_be_inf(c, tainted, memo)
    memo[e] = r
    return r
