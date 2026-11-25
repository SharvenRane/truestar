/-!
# T9: fp8 decoder facts, all 256 codes checked by `decide`

Values are returned exactly as integers in units of `2^-16` (the smallest subnormals are
e4m3fn `2^-9`, e4m3fnuz `2^-10`, e5m2 `2^-16`), with NaN and the e5m2 infinities as their own
cases.

* e4m3fn (OCP FP8, `torch.float8_e4m3fn`): bias 7, no infinities, NaN = `S.1111.111`.
* e4m3fnuz (`torch.float8_e4m3fnuz`): bias 8, no infinities, no negative zero, NaN = `0x80` only.
* e5m2 (OCP FP8, IEEE style): bias 15, infinities at exponent 31 with fraction 0, NaN otherwise.

`TcCheck.lean` also compares these decoders with torch's float8 dtypes on all 256 codes (fixture).
-/
namespace Truestar.Fp8

inductive Val where
  | num : Int → Val    -- value in units of 2^-16
  | inf : Bool → Val   -- true = negative
  | nan : Val
deriving Repr, DecidableEq

def signed (neg : Bool) (v : Int) : Int := if neg then -v else v

/-- OCP e4m3fn. -/
def e4m3fn (c : Fin 256) : Val :=
  let neg := c.val ≥ 128
  let e := (c.val / 8) % 16
  let m := c.val % 8
  if e = 15 ∧ m = 7 then .nan
  else if e = 0 then .num (signed neg (m * 2 ^ 7))          -- m * 2^-9
  else .num (signed neg ((8 + m) * 2 ^ (e + 6)))             -- (8+m) * 2^(e-10)

/-- e4m3fnuz: bias 8, NaN at 0x80, no negative zero. -/
def e4m3fnuz (c : Fin 256) : Val :=
  let neg := c.val ≥ 128
  let e := (c.val / 8) % 16
  let m := c.val % 8
  if c.val = 128 then .nan
  else if e = 0 then .num (signed neg (m * 2 ^ 6))          -- m * 2^-10
  else .num (signed neg ((8 + m) * 2 ^ (e + 5)))             -- (8+m) * 2^(e-11)

/-- OCP e5m2. -/
def e5m2 (c : Fin 256) : Val :=
  let neg := c.val ≥ 128
  let e := (c.val / 4) % 32
  let m := c.val % 4
  if e = 31 then (if m = 0 then .inf neg else .nan)
  else if e = 0 then .num (signed neg m)                     -- m * 2^-16
  else .num (signed neg ((4 + m) * 2 ^ (e - 1)))             -- (4+m) * 2^(e-17)

def isNan (v : Val) : Bool := match v with | .nan => true | _ => false

def numOr0 (v : Val) : Int := match v with | .num x => x | _ => 0

/-- NaN codes of e4m3fn: exactly 0x7F and 0xFF. -/
theorem e4m3fn_nan : ∀ c : Fin 256, isNan (e4m3fn c) = (c.val = 0x7F || c.val = 0xFF) := by
  decide +kernel

/-- NaN code of e4m3fnuz: exactly 0x80. -/
theorem e4m3fnuz_nan : ∀ c : Fin 256, isNan (e4m3fnuz c) = (c.val = 0x80) := by
  decide +kernel

/-- e4m3fn largest finite magnitude: 448 (`448 * 2^16` units), reached at 0x7E and 0xFE. -/
theorem e4m3fn_max : (∀ c : Fin 256, (numOr0 (e4m3fn c)).natAbs ≤ 448 * 2 ^ 16) ∧
    e4m3fn 0x7E = .num (448 * 2 ^ 16) ∧ e4m3fn 0xFE = .num (-(448 * 2 ^ 16)) := by
  decide +kernel

/-- e4m3fnuz largest finite magnitude: 240, reached at 0x7F and 0xFF. -/
theorem e4m3fnuz_max : (∀ c : Fin 256, (numOr0 (e4m3fnuz c)).natAbs ≤ 240 * 2 ^ 16) ∧
    e4m3fnuz 0x7F = .num (240 * 2 ^ 16) ∧ e4m3fnuz 0xFF = .num (-(240 * 2 ^ 16)) := by
  decide +kernel

/-- e5m2 largest finite magnitude: 57344. -/
theorem e5m2_max : (∀ c : Fin 256, (numOr0 (e5m2 c)).natAbs ≤ 57344 * 2 ^ 16) ∧
    e5m2 0x7B = .num (57344 * 2 ^ 16) := by
  decide +kernel

/-- The same bits mean half the value in e4m3fnuz, except at the three special codes (0x7F and
0xFF are NaN in e4m3fn and 240 in e4m3fnuz; 0x80 is negative zero in e4m3fn and NaN in
e4m3fnuz). Reading one format's bytes as the other is therefore never a silent rescale only. -/
def halfOf (a b : Val) : Bool :=
  match a, b with
  | .num x, .num y => y == 2 * x
  | _, _ => false

theorem fnuz_half : ∀ c : Fin 256, c.val ≠ 0x7F → c.val ≠ 0xFF → c.val ≠ 0x80 →
    halfOf (e4m3fnuz c) (e4m3fn c) = true := by
  decide +kernel

/-- Positive e4m3fn codes are strictly increasing in value (0x00 to 0x7E). -/
theorem e4m3fn_monotone : ∀ c : Fin 126, numOr0 (e4m3fn ⟨c.val, by omega⟩) <
    numOr0 (e4m3fn ⟨c.val + 1, by omega⟩) := by
  decide +kernel

end Truestar.Fp8
