import Lean.Data.Json
import TruestarProofs.TCWindow
import TruestarProofs.Fp8

/-!
Cross check: run the Lean accumulator model (`Truestar.TC.group` / `chain`, the functions T1 is
proved about) on the probe vectors exported from the Python simulator
(`scripts/gen_lean_tc_fixture.py` -> `fixtures/tc_probes.json`) and compare every result bit for
bit; also compare the T9 fp8 decoders with torch's float8 dtypes on all 256 codes.

Usage: `lake exe tccheck [fixtures/tc_probes.json]`; exit code 1 on any mismatch.
-/
open Lean Truestar.TC

/-- Round a value (units of `2^-L`) once to float32: truncation or nearest even, with the
float32 subnormal floor; overflow is not modelled (no probe reaches it). -/
def roundF32 (L : Nat) (mode : String) (v : Int) : Int :=
  if v = 0 then 0 else
  let e : Int := max (Nat.log2 v.natAbs : Int) ((L : Int) - 126)
  let s : Int := e - 23
  if s ≤ 0 then v else
  let d : Int := 2 ^ s.toNat
  let q := if mode = "rne" then
      let fl := v / d
      let r := v - fl * d
      if 2 * r > d then fl + 1 else if 2 * r < d then fl else (if fl % 2 = 0 then fl else fl + 1)
    else v.tdiv d
  q * d

/-- The exponent the unit reads from a float32 accumulator value (units). -/
def cexpF32 (L : Nat) (c : Int) : Int := max (Nat.log2 c.natAbs : Int) ((L : Int) - 126)

def chunks {α : Type} (n : Nat) (l : List α) : List (List α) :=
  go l.length l
where
  go : Nat → List α → List (List α)
  | 0, _ => []
  | _, [] => []
  | fuel + 1, l => l.take n :: go fuel (l.drop n)

def getStrInt (j : Json) : Except String Int := do
  let s ← j.getStr?
  match s.toInt? with
  | some v => pure v
  | none => throw s!"not an integer: {s}"

def checkProbe (L : Nat) (p : Json) : Except String (String × Bool × Bool) := do
  let fam ← (p.getObjValD "family").getStr?
  let u := p.getObjValD "unit"
  let name ← (u.getObjValD "name").getStr?
  let F ← (u.getObjValD "F").getNat?
  let grp ← (u.getObjValD "group").getNat?
  let cutS ← (u.getObjValD "cut").getStr?
  let final ← (u.getObjValD "final").getStr?
  let floorJ := u.getObjValD "floor"
  let floor : Option Int := match floorJ.getInt? with
    | .ok f => some f
    | .error _ => none
  let unit : MatrixUnit := ⟨F, if cutS = "rd" then .rd else .rz, floor⟩
  let c ← getStrInt (p.getObjValD "c")
  let termsJ ← (p.getObjValD "terms").getArr?
  let terms ← termsJ.toList.mapM fun t => do
    let a ← t.getArr?
    let v ← getStrInt a[0]!
    let e ← a[1]!.getInt?
    pure (Term.mk v e)
  let expected ← getStrInt (p.getObjValD "expected")
  let got := chain unit (roundF32 L final) (cexpF32 L) c (chunks grp terms)
  let exactOk ← match p.getObjValD "exact" with
    | .null => pure true
    | j => do let x ← getStrInt j; pure (if fam = "digits" then got = x else true)
  pure (s!"{fam} {name}", got = expected, exactOk)

def fp8Val (j : Json) : Except String Truestar.Fp8.Val := do
  let s ← j.getStr?
  if s = "nan" then pure .nan
  else if s = "inf" then pure (.inf false)
  else if s = "-inf" then pure (.inf true)
  else match s.toInt? with
    | some v => pure (.num v)
    | none => throw s!"bad fp8 value {s}"

def main (args : List String) : IO UInt32 := do
  let path := args.headD "fixtures/tc_probes.json"
  let txt ← IO.FS.readFile path
  let j ← IO.ofExcept (Json.parse txt)
  let L ← IO.ofExcept ((j.getObjValD "L").getNat?)
  let probes ← IO.ofExcept ((j.getObjValD "probes").getArr?)
  let mut bad := 0
  let mut badExact := 0
  let mut n := 0
  for p in probes do
    let (tag, ok, okx) ← IO.ofExcept (checkProbe L p)
    n := n + 1
    if !ok then
      bad := bad + 1
      IO.println s!"MISMATCH probe {n - 1} ({tag})"
    if !okx then
      badExact := badExact + 1
      IO.println s!"NOT EXACT probe {n - 1} ({tag})"
  let fp8 ← IO.ofExcept ((j.getObjValD "fp8").getArr?)
  let mut bad8 := 0
  for r in fp8 do
    let code ← IO.ofExcept ((r.getObjValD "code").getNat?)
    let c : Fin 256 := ⟨code % 256, Nat.mod_lt _ (by decide)⟩
    let a ← IO.ofExcept (fp8Val (r.getObjValD "e4m3fn"))
    let b ← IO.ofExcept (fp8Val (r.getObjValD "e4m3fnuz"))
    let d ← IO.ofExcept (fp8Val (r.getObjValD "e5m2"))
    -- torch decodes the e4m3fnuz code 0x00 and e4m3fn code 0x80 as +0 and -0: both are 0 here
    if a != Truestar.Fp8.e4m3fn c || b != Truestar.Fp8.e4m3fnuz c || d != Truestar.Fp8.e5m2 c then
      bad8 := bad8 + 1
      IO.println s!"FP8 MISMATCH code {code}"
  IO.println s!"tensor core probes: {n}, mismatches: {bad}, digit chains not exact: {badExact}"
  IO.println s!"fp8 codes: {fp8.size}, mismatches: {bad8}"
  return (if bad + badExact + bad8 = 0 then 0 else 1)
