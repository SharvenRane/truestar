"""C golden vectors for a microcontroller test harness (docs/INTREF.md, gap item 13).

`emit_header` writes one self contained C99 header: per layer the integer input(s), the expected
output under the chosen convention, a tolerance in LSB, an FNV-1a checksum of the expected bytes
and the quantization parameters, plus `ts_golden_check`, which a board's harness calls with the
output its kernel produced. No allocation, no stdio, only <stdint.h>.

Tolerance: `exact` gives 0 LSB (the board must use this convention); `portable` gives, per layer,
the largest difference between this convention and any other documented one on the same layer
input, so a board passes whichever documented library it links, and fails beyond that.

`parse_header` reads a header back and checks every array, length, checksum and tolerance, which
is how the header is validated on a machine without a C compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .conventions import CONVENTIONS, documented_conventions
from .ops import run_op


@dataclass
class GoldenLayer:
    name: str
    op: str
    inputs: Dict[str, np.ndarray]  # integer activations fed to the layer (a, b or x)
    params: dict
    weights: Dict[str, np.ndarray] = field(default_factory=dict)  # w, b: part of the deployed model
    expected: Optional[np.ndarray] = None
    tol_lsb: int = 0
    spread: Dict[str, int] = field(default_factory=dict)  # max LSB vs each other convention


def fnv1a(data: np.ndarray) -> int:
    h = 0x811C9DC5
    for byte in np.ascontiguousarray(data).tobytes():
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def demo_network(seed: int = 0):
    """A small int8 network covering conv, max pool, avg pool, fully connected, add, mul and
    softmax, with seeded weights and scales. Returns (input, list of layer specs)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (1, 3, 8, 8)).astype(np.int8)
    specs = [
        dict(name="conv1", op="conv", weights={"w": rng.integers(-127, 128, (4, 3, 3, 3)).astype(np.int8),
                                               "b": rng.integers(-3000, 3000, 4).astype(np.int32)},
             params=dict(x_zp=-3, sx=0.0241, sw=rng.uniform(0.004, 0.012, 4), sy=0.0917, y_zp=-128,
                         padding=1, act_min=-128)),
        dict(name="pool1", op="max_pool", params=dict(kernel=2, stride=2)),
        dict(name="pool2", op="avg_pool", params=dict(kernel=3, stride=1, padding=1)),
        dict(name="fc1", op="fully_connected",
             weights={"w": rng.integers(-127, 128, (10, 64)).astype(np.int8),
                      "b": rng.integers(-5000, 5000, 10).astype(np.int32)},
             params=dict(x_zp=-128, sx=0.0917, sw=0.0061, sy=0.173, y_zp=3)),
        dict(name="add1", op="add", operand=rng.integers(-128, 128, (1, 10)).astype(np.int8),
             params=dict(a_zp=3, sa=0.173, b_zp=-7, sb=0.061, sy=0.197, y_zp=-2)),
        dict(name="mul1", op="mul", operand=rng.integers(-128, 128, (1, 10)).astype(np.int8),
             params=dict(a_zp=-2, sa=0.197, b_zp=11, sb=0.0123, sy=0.0331, y_zp=5)),
        dict(name="softmax", op="softmax", params=dict(x_zp=5, sx=0.0331, sy=1.0 / 256, y_zp=-128)),
    ]
    return x, specs


def _layer_inputs(spec: dict, act: np.ndarray) -> Dict[str, np.ndarray]:
    if spec["op"] in ("add", "mul"):
        return {"a": act, "b": spec["operand"]}
    return {"x": act}


def build_layers(x: np.ndarray, specs: Sequence[dict], convention: str, tolerance: str = "exact",
                 others: Optional[Sequence[str]] = None) -> List[GoldenLayer]:
    """Run the chain under `convention`; each layer's input is the previous expected output."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}")
    if tolerance not in ("exact", "portable"):
        raise ValueError("tolerance is 'exact' or 'portable'")
    others = [c for c in (others or documented_conventions()) if c != convention]
    act = np.asarray(x)
    layers = []
    for spec in specs:
        if spec["op"] == "fully_connected" and act.ndim != 2:
            act = act.reshape(act.shape[0], -1)
        inputs = _layer_inputs(spec, act)
        allin = {**inputs, **spec.get("weights", {})}
        exp = run_op(spec["op"], allin, spec["params"], convention).astype(act.dtype)
        spread = {c: int(np.abs(run_op(spec["op"], allin, spec["params"], c) - exp).max())
                  for c in others}
        tol = max(spread.values(), default=0) if tolerance == "portable" else 0
        layers.append(GoldenLayer(spec["name"], spec["op"], inputs, spec["params"],
                                  spec.get("weights", {}), exp, tol, spread))
        act = exp
    return layers


def _c_array(ctype: str, name: str, a: np.ndarray, per_line: int = 16) -> str:
    vals = [str(int(v)) for v in np.asarray(a).ravel()]
    rows = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const {ctype} {name}[{len(vals)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"


def _ctype(a: np.ndarray) -> str:
    return {np.dtype(np.int8): "int8_t", np.dtype(np.int16): "int16_t"}[np.asarray(a).dtype]


_HELPERS = r"""
static inline uint32_t ts_golden_fnv1a(const void *p, int32_t n_bytes) {
  const uint8_t *b = (const uint8_t *)p;
  uint32_t h = 2166136261u;
  for (int32_t i = 0; i < n_bytes; ++i) { h ^= b[i]; h *= 16777619u; }
  return h;
}

/* Elements of `got` (same type and length as L->expected) more than L->tol_lsb away from the
   expected output; the largest difference goes to *worst when worst is not NULL. 0 means pass. */
static inline int32_t ts_golden_check(const ts_golden_layer *L, const void *got, int32_t *worst) {
  int32_t bad = 0, w = 0;
  for (int32_t i = 0; i < L->n_out; ++i) {
    int32_t e = L->bits == 16 ? ((const int16_t *)L->expected)[i] : ((const int8_t *)L->expected)[i];
    int32_t g = L->bits == 16 ? ((const int16_t *)got)[i] : ((const int8_t *)got)[i];
    int32_t d = e > g ? e - g : g - e;
    if (d > w) w = d;
    if (d > L->tol_lsb) ++bad;
  }
  if (worst) *worst = w;
  return bad;
}

/* 1 if the expected bytes of every layer still hash to their recorded checksum. */
static inline int ts_golden_intact(void) {
  for (int32_t i = 0; i < TS_GOLDEN_LAYERS; ++i) {
    const ts_golden_layer *L = &ts_golden[i];
    if (ts_golden_fnv1a(L->expected, L->n_out * (L->bits / 8)) != L->fnv1a) return 0;
  }
  return 1;
}
"""


def emit_header(layers: Sequence[GoldenLayer], convention: str, tolerance: str,
                guard: str = "TS_GOLDEN_H") -> str:
    out = [f"/* Truestar integer reference golden vectors. Convention: {convention} "
           f"({CONVENTIONS[convention].description}). Tolerance: {tolerance}.",
           "   Generated by `truestar intref golden`; layer inputs are the previous layer's expected",
           "   output, so each layer can be checked alone. Layouts: NCHW, weights OIHW. */",
           f"#ifndef {guard}", f"#define {guard}", "#include <stdint.h>", "",
           f'#define TS_GOLDEN_CONVENTION "{convention}"',
           f"#define TS_GOLDEN_LAYERS {len(layers)}", "",
           "typedef struct {", "  const char *name;", "  const char *op;", "  int32_t bits;",
           "  int32_t n_in;", "  const void *in;", "  const void *in2; /* second operand (add, mul) or 0 */",
           "  int32_t n_out;", "  const void *expected;", "  int32_t tol_lsb;", "  uint32_t fnv1a;",
           "  int32_t in_zp;", "  int32_t out_zp;", "  float out_scale;", "} ts_golden_layer;", ""]
    entries = []
    for i, L in enumerate(layers):
        main = L.inputs.get("x", L.inputs.get("a"))
        ct = _ctype(L.expected)
        out.append(f"/* layer {i}: {L.name} ({L.op}), input shape {list(main.shape)}, output "
                   f"shape {list(L.expected.shape)}, max LSB vs other conventions: "
                   + (", ".join(f"{k} {v}" for k, v in L.spread.items()) or "none") + " */")
        out.append(_c_array(_ctype(main), f"ts_l{i}_in", main))
        in2 = "0"
        if "b" in L.inputs and L.op in ("add", "mul"):
            out.append(_c_array(_ctype(L.inputs["b"]), f"ts_l{i}_in2", L.inputs["b"]))
            in2 = f"ts_l{i}_in2"
        out.append(_c_array(ct, f"ts_l{i}_expected", L.expected))
        p = L.params
        in_zp = int(p.get("x_zp", p.get("a_zp", 0)))
        out_zp = int(p.get("y_zp", in_zp if L.op in ("max_pool", "avg_pool") else 0))
        scale = float(p.get("sy", p.get("sx", 1.0)))
        entries.append(f'  {{"{L.name}", "{L.op}", {16 if ct == "int16_t" else 8}, {main.size}, '
                       f"ts_l{i}_in, {in2}, {L.expected.size}, ts_l{i}_expected, {L.tol_lsb}, "
                       f"0x{fnv1a(L.expected):08X}u, {in_zp}, {out_zp}, {scale!r}f}}")
    out.append("static const ts_golden_layer ts_golden[TS_GOLDEN_LAYERS] = {\n" + ",\n".join(entries) + "\n};")
    out.append(_HELPERS)
    out.append(f"#endif /* {guard} */\n")
    return "\n".join(out)


_ARRAY = re.compile(r"static const (int8_t|int16_t) (\w+)\[(\d+)\] = \{([^}]*)\};")
_ENTRY = re.compile(r'\{"(\w+)", "(\w+)", (\d+), (\d+), (\w+), (\w+), (\d+), (\w+), (\d+), '
                    r"0x([0-9A-F]{8})u, (-?\d+), (-?\d+), ([-+0-9.eE]+)f\}")


def parse_header(text: str) -> dict:
    """Parse a golden header back and check it is internally consistent: declared and actual
    lengths, element ranges, checksums, layer count. Raises ValueError listing every problem."""
    problems: List[str] = []
    arrays: Dict[str, np.ndarray] = {}
    for ctype, name, n, body in _ARRAY.findall(text):
        vals = np.array([int(v) for v in body.replace("\n", " ").split(",") if v.strip()],
                        dtype=np.int64)
        lo, hi = (-128, 127) if ctype == "int8_t" else (-32768, 32767)
        if vals.size != int(n):
            problems.append(f"{name}: declared {n} elements, holds {vals.size}")
        if vals.size and (vals.min() < lo or vals.max() > hi):
            problems.append(f"{name}: value outside {ctype}")
        arrays[name] = vals.astype(np.int8 if ctype == "int8_t" else np.int16)
    m = re.search(r"#define TS_GOLDEN_LAYERS (\d+)", text)
    layers = []
    for e in _ENTRY.findall(text):
        name, op, bits, n_in, a_in, a_in2, n_out, a_exp, tol, h, in_zp, out_zp, scale = e
        exp = arrays.get(a_exp)
        if exp is None:
            problems.append(f"{name}: expected array {a_exp} missing")
            continue
        if exp.size != int(n_out):
            problems.append(f"{name}: n_out {n_out} but {a_exp} has {exp.size}")
        if arrays.get(a_in) is None or arrays[a_in].size != int(n_in):
            problems.append(f"{name}: input array {a_in} missing or wrong length")
        if fnv1a(exp) != int(h, 16):
            problems.append(f"{name}: checksum 0x{h} does not match the expected bytes")
        layers.append({"name": name, "op": op, "bits": int(bits), "input": arrays.get(a_in),
                       "input2": arrays.get(a_in2) if a_in2 != "0" else None, "expected": exp,
                       "tol_lsb": int(tol), "in_zp": int(in_zp), "out_zp": int(out_zp),
                       "out_scale": float(scale)})
    if not m or int(m.group(1)) != len(layers):
        problems.append(f"TS_GOLDEN_LAYERS {m.group(1) if m else 'missing'} but {len(layers)} entries")
    if problems:
        raise ValueError("golden header invalid: " + "; ".join(problems))
    conv = re.search(r'#define TS_GOLDEN_CONVENTION "(\w+)"', text)
    return {"convention": conv.group(1) if conv else None, "layers": layers}


def verify_roundtrip(layers: Sequence[GoldenLayer], text: str) -> List[str]:
    """Differences between the layers written and the header parsed back (empty when equal)."""
    parsed = parse_header(text)["layers"]
    bad = []
    if len(parsed) != len(layers):
        return [f"{len(parsed)} layers parsed, {len(layers)} written"]
    for L, P in zip(layers, parsed):
        main = L.inputs.get("x", L.inputs.get("a"))
        if not np.array_equal(P["expected"], np.asarray(L.expected).ravel()):
            bad.append(f"{L.name}: expected output differs")
        if not np.array_equal(P["input"], np.asarray(main).ravel()):
            bad.append(f"{L.name}: input differs")
        if P["tol_lsb"] != L.tol_lsb:
            bad.append(f"{L.name}: tolerance differs")
    return bad


def check_outputs(parsed: dict, outputs: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """What `ts_golden_check` returns on a board, computed in Python: per layer, elements over
    tolerance and the worst difference (used to test the header logic without a compiler)."""
    res = {}
    for L in parsed["layers"]:
        if L["name"] not in outputs:
            continue
        d = np.abs(np.asarray(outputs[L["name"]], np.int64).ravel() - L["expected"].astype(np.int64))
        res[L["name"]] = {"bad": int((d > L["tol_lsb"]).sum()), "worst": int(d.max(initial=0))}
    return res
