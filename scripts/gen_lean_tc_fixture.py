"""Export probe vectors of the tensor core simulator for the Lean cross check.

Writes proofs/lean/fixtures/tc_probes.json. Each probe is one output element of a dot product
``c + sum_k a_k b_k`` on one matrix unit model of src/truestar/devsim/tensorcore.py, with the
products already split into (value, exponent the unit reads), exactly as ``unit_matmul`` forms
them, and the simulator's result. ``lake exe tccheck`` (proofs/lean/TcCheck.lean) recomputes every
probe with the Lean model of TruestarProofs/TCWindow.lean (``group`` / ``chain``) and compares.

All values are exact integers in units of ``2**-L`` (L = 160), written as decimal strings.

Probe families:
* xchip: the cross chip probe (1.0 plus 15 products of 2**-k, large product first or last);
* random: normal values with wide exponent spreads in fp16, bf16 and tf32, random c;
* digits: the fp16 engine of tcexact (digit products chained from C = 1.5 * 2**23), worst cases
  included, on every generation;
* variants: the same random vectors on units with a 10 bit window, ``rd`` cut and ``rne`` final
  rounding, so the code paths no shipped unit uses are compared too.

Also writes all 256 fp8 codes decoded by torch (float8_e4m3fn, float8_e4m3fnuz, float8_e5m2) in
units of 2**-16 for the T9 decoder check.

Usage: python scripts/gen_lean_tc_fixture.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from truestar.devsim.tensorcore import FORMATS, GENERATIONS, split, unit_matmul

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "proofs" / "lean" / "fixtures" / "tc_probes.json"
L = 160


def units(x: float) -> str:
    q = Fraction(float(x)) * (1 << L)
    assert q.denominator == 1, x
    return str(q.numerator)


def unit_json(u) -> dict:
    return {"name": u.name, "F": u.frac_bits, "group": u.group, "cut": u.cut, "final": u.final,
            "floor": None if u.floor_exp is None else u.floor_exp + L}


def probe(u, fmt: str, a: np.ndarray, b: np.ndarray, c: float, family: str) -> dict:
    """One output: a [K], b [K], scalar c."""
    f = FORMATS[fmt]
    an, as_, ae = split(a, f)
    bn, bs, be = split(b, f)
    terms = []
    for k in range(len(a)):
        sig = int(as_[k]) * int(bs[k])
        sign = -1 if bool(an[k]) != bool(bn[k]) else 1
        e = int(ae[k]) + int(be[k])          # the exponent the unit reads (unnormalized)
        shift = e - 2 * f.man + L            # value = sig * 2**(e - 2 man)
        assert shift >= 0
        v = sig << shift
        terms.append([str(v if sign > 0 else -v), e + L])
    out = unit_matmul(a[None, :], b[:, None], np.array([[c]]), u, fmt)[0, 0]
    return {"family": family, "unit": unit_json(u), "c": units(c), "terms": terms,
            "expected": units(out)}


def chain_probe(u, da: list, db: list, family: str) -> dict:
    """The fp16 engine: chained digit planes from C, as one long chain (K a multiple of group)."""
    f = FORMATS["fp16"]
    terms = []
    c = np.full((1, 1), 12582912.0)
    for x, y in zip(da, db):
        an, as_, ae = split(x, f)
        bn, bs, be = split(y, f)
        for k in range(len(x)):
            sig = int(as_[k]) * int(bs[k])
            neg = bool(an[k]) != bool(bn[k])
            e = int(ae[k]) + int(be[k])
            v = sig << (e - 2 * f.man + L)
            terms.append([str(-v if neg else v), e + L])
        assert len(x) % u.group == 0
        c = unit_matmul(x[None, :], y[:, None], c, u, "fp16")
    return {"family": family, "unit": unit_json(u), "c": units(12582912.0), "terms": terms,
            "expected": units(c[0, 0]), "exact": str((12582912 + sum(
                int(p) * int(q) for x, y in zip(da, db) for p, q in zip(x, y))) << L)}


def representable(x: np.ndarray, fmt: str) -> np.ndarray:
    import torch
    t = torch.from_numpy(x.astype(np.float32))
    if fmt == "fp16":
        return t.to(torch.float16).float().numpy().astype(np.float64)
    if fmt == "bf16":
        return t.to(torch.bfloat16).float().numpy().astype(np.float64)
    # tf32: keep 10 fraction bits (round to nearest even on the bits)
    u = t.view(torch.int32)
    u = (u + 4095 + ((u >> 13) & 1)) & -8192
    return u.view(torch.float32).numpy().astype(np.float64)


def fp8_table() -> list:
    import torch
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    rows = []
    cols = {}
    for name, dt in (("e4m3fn", torch.float8_e4m3fn), ("e4m3fnuz", torch.float8_e4m3fnuz),
                     ("e5m2", torch.float8_e5m2)):
        cols[name] = codes.view(dt).float().tolist()
    for i in range(256):
        row = {"code": i}
        for name, vals in cols.items():
            v = vals[i]
            if v != v:
                row[name] = "nan"
            elif v in (float("inf"), float("-inf")):
                row[name] = "inf" if v > 0 else "-inf"
            else:
                q = Fraction(v) * (1 << 16)
                assert q.denominator == 1
                row[name] = str(q.numerator)
        rows.append(row)
    return rows


def main() -> int:
    probes = []
    rs = np.random.RandomState(20261125)
    gens = ("v100", "a100", "ada", "h100", "b200", "sm120")
    # xchip probe
    for gen in gens:
        for fmt in ("fp16", "bf16", "tf32"):
            if fmt not in GENERATIONS[gen]:
                continue
            u = GENERATIONS[gen][fmt]
            for k in range(18, 34):
                h1, h2 = k // 2, k - k // 2
                for last in (False, True):
                    a = np.full(16, 2.0 ** -h1)
                    b = np.full(16, 2.0 ** -h2)
                    j = 15 if last else 0
                    a[j], b[j] = 1.0, 1.0
                    probes.append(probe(u, fmt, a, b, 0.0, "xchip"))
    # random
    for gen in gens:
        for fmt in ("fp16", "bf16", "tf32"):
            if fmt not in GENERATIONS[gen]:
                continue
            u = GENERATIONS[gen][fmt]
            for _ in range(8):
                kk = int(rs.choice([16, 32, 64]))
                spread = 6 if fmt == "fp16" else 20
                a = rs.standard_normal(kk) * 2.0 ** rs.randint(-spread, spread + 1, size=kk)
                b = rs.standard_normal(kk) * 2.0 ** rs.randint(-spread, spread + 1, size=kk)
                a, b = representable(a, fmt), representable(b, fmt)
                c = float(np.float32(rs.standard_normal() * 2.0 ** rs.randint(-8, 9)))
                if rs.rand() < 0.25:
                    c = 0.0
                probes.append(probe(u, fmt, a, b, c, "random"))
    # tcexact fp16 engine digit chains
    for gen in gens:
        u = GENERATIONS[gen]["fp16"]
        for kb in (16, 64, 128):
            planes = []
            worst_a = [np.full(kb, 127.0), np.full(kb, -64.0), np.full(kb, 64.0)]
            worst_b = [np.full(kb, 127.0), np.full(kb, 64.0), np.full(kb, -64.0)]
            rnd_a = [rs.randint(-64, 65, size=kb).astype(np.float64) for _ in range(3)]
            rnd_b = [rs.randint(-64, 65, size=kb).astype(np.float64) for _ in range(3)]
            rnd_a[0] = rs.randint(-127, 128, size=kb).astype(np.float64)
            rnd_b[0] = rs.randint(-127, 128, size=kb).astype(np.float64)
            for da, db in ((worst_a, worst_b), (rnd_a, rnd_b)):
                for pairs in (((0, 0),), ((0, 1), (1, 0)), ((0, 2), (1, 1), (2, 0))):
                    planes.append(([da[p] for p, _ in pairs], [db[q] for _, q in pairs]))
            for xa, xb in planes:
                probes.append(chain_probe(u, xa, xb, "digits"))
    # variants
    base = GENERATIONS["ada"]["fp16"]
    variants = (replace(base, frac_bits=10, name="ada:fp16/w10"),
                replace(base, cut="rd", name="ada:fp16/rd"),
                replace(base, final="rne", name="ada:fp16/rne"),
                replace(base, cut="rd", final="rne", frac_bits=12, floor_exp=None, name="ada:fp16/rd-rne-w12"))
    for u in variants:
        for _ in range(12):
            kk = int(rs.choice([16, 32]))
            a = representable(rs.standard_normal(kk) * 2.0 ** rs.randint(-6, 7, size=kk), "fp16")
            b = representable(rs.standard_normal(kk) * 2.0 ** rs.randint(-6, 7, size=kk), "fp16")
            c = float(np.float32(rs.standard_normal()))
            probes.append(probe(u, "fp16", a, b, c, "variants"))
        da = [np.where(np.arange(64) % 2 == 0, 64.0, 1.0)]
        db = [np.full(64, 63.0)]
        probes.append(chain_probe(u, da, db, "variants"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"L": L, "probes": probes, "fp8": fp8_table()}, separators=(",", ":")) + "\n",
                   encoding="utf-8", newline="\n")
    fams = {}
    for p in probes:
        fams[p["family"]] = fams.get(p["family"], 0) + 1
    print(f"wrote {OUT}: {len(probes)} probes {fams}, 256 fp8 codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
