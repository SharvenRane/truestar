import TruestarProofs.Basic

/-!
# T3: the digit split

`tcexact.py` step 3: `t = a * 2^(7 - ea)` with `|t| < 128 / (1 + 2^-7) < 127.01`, then
`d0 = rn(t)`, `t1 = (t - d0) * 128`, `d1 = rn(t1)`, `t2 = (t1 - d1) * 128`, `d2 = rn(t2)`, where
`rn` is round to the nearest integer, ties to even.

Every finite float32 value is an integer multiple of `2^-149`, so `t` is written exactly as
`T / S` with `T` an integer and `S = 2^149`; all the steps are then integer arithmetic.

What is proved here (exact arithmetic):
* the reconstruction identity `t = d0 + d1 2^-7 + d2 2^-14 + R 2^-14 / S` with the dropped part
  `|R| <= S / 2` (at most `2^-15` in units of `t`, i.e. `2^(ea - 22)` in units of `a`);
* the dropped part is zero, so the three digits reconstruct `t` exactly, whenever `t` is a
  multiple of `2^-14` (float16 values within `2^10` of the block maximum, for example);
* the digit ranges `|d0| <= 127`, `|d1|, |d2| <= 64` (int8 holds them, and float16 holds them
  exactly) for every `|t| < 127.5`;
* the magic constant: `rn(t) + 1.5 * 2^23 = rn(t + 1.5 * 2^23)` (the float32 addition
  `t + 1.5 * 2^23` lands in `[2^23, 2^24)` where float32 round to nearest even is exactly `rn`),
  and the float32 bit pattern `0x4B400000 + i` decodes to `1.5 * 2^23 + i` for `|i| < 2^22`;
* the integer sums of one K block of up to `MAX_KB = 128` columns stay at most `2,605,056`,
  below `2^22` (no int32 overflow; exact float32 conversion).

Assumed, not proved here: that each float32 operation of the kernel (`t - d0`, `* 128`, the scale
by `2^(7 - ea)`) is exact. They are (Sterbenz style subtraction, power of two scaling inside the
normal range, which the exponent clamp of `tcexact.py` ensures); `tests/test_tcexact.py` checks
the float32 kernel against these integer results.
-/
namespace Truestar.Digits

/-- `2^149`: every finite float32 value is an integer multiple of `1 / S`. -/
local notation "S" => (713623846352979940529142984724747568191373312 : Int)

/-- Round `T / S` to the nearest integer, ties to even. -/
def rn (T : Int) : Int :=
  if (T + S / 2) % S = 0 ∧ (T + S / 2) / S % 2 ≠ 0 then (T + S / 2) / S - 1 else (T + S / 2) / S

theorem rn_err (T : Int) : -(S / 2) ≤ T - rn T * S ∧ T - rn T * S ≤ S / 2 := by
  unfold rn; split <;> omega

/-- `rn` of an exact multiple of `S` is that multiple. -/
theorem rn_exact (m : Int) : rn (m * S) = m := by
  unfold rn; split <;> omega

/-- The digits and the dropped remainder `R` (in units of `1 / S`, after two scalings by 128). -/
def digits (T : Int) : Int × Int × Int × Int :=
  let d0 := rn T
  let T1 := (T - d0 * S) * 128
  let d1 := rn T1
  let T2 := (T1 - d1 * S) * 128
  let d2 := rn T2
  (d0, d1, d2, T2 - d2 * S)

/-- Reconstruction identity: `T * 2^14 = (d0 2^14 + d1 2^7 + d2) S + R`, with `|R| <= S / 2`. -/
theorem digits_reconstruct (T : Int) :
    let (d0, d1, d2, R) := digits T
    T * 16384 = (d0 * 16384 + d1 * 128 + d2) * S + R ∧ -(S / 2) ≤ R ∧ R ≤ S / 2 := by
  simp only [digits]
  have := rn_err ((((T - rn T * S) * 128) - rn ((T - rn T * S) * 128) * S) * 128)
  refine ⟨?_, this.1, this.2⟩
  simp only [Int.add_mul]
  omega

/-- The digits reconstruct `t` exactly when `t` is a multiple of `2^-14`. -/
theorem digits_exact_of_fits (T : Int) (h : (2 : Int) ^ 135 ∣ T) :
    let (d0, d1, d2, R) := digits T
    R = 0 ∧ T * 16384 = (d0 * 16384 + d1 * 128 + d2) * S := by
  obtain ⟨k, rfl⟩ := h
  simp only [digits]
  have e : ((2 ^ 135 * k - rn (2 ^ 135 * k) * S) * 128 -
      rn ((2 ^ 135 * k - rn (2 ^ 135 * k) * S) * 128) * S) * 128 =
      (k - rn (2 ^ 135 * k) * 16384 - rn ((2 ^ 135 * k - rn (2 ^ 135 * k) * S) * 128) * 128) * S := by
    simp only [Int.sub_mul, Int.mul_assoc]; omega
  rw [e, rn_exact]
  constructor
  · omega
  · simp only [Int.add_mul, Int.sub_mul, Int.mul_assoc]; omega

/-- Digit ranges: for `|t| < 127.5`, `|d0| <= 127` and `|d1|, |d2| <= 64`. -/
theorem digits_bounds (T : Int) (h1 : -255 * S < 2 * T) (h2 : 2 * T < 255 * S) :
    let (d0, d1, d2, _R) := digits T
    (-127 ≤ d0) ∧ d0 ≤ 127 ∧ -64 ≤ d1 ∧ d1 ≤ 64 ∧ -64 ≤ d2 ∧ d2 ≤ 64 := by
  simp only [digits]
  have e0 := rn_err T
  have e1 := rn_err ((T - rn T * S) * 128)
  have e2 := rn_err (((T - rn T * S) * 128 - rn ((T - rn T * S) * 128) * S) * 128)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The int8 range used: every digit lies in `[-128, 127]`. -/
theorem digits_int8 (T : Int) (h1 : -255 * S < 2 * T) (h2 : 2 * T < 255 * S) :
    let (d0, d1, d2, _R) := digits T
    (-128 ≤ d0) ∧ d0 ≤ 127 ∧ -128 ≤ d1 ∧ d1 ≤ 127 ∧ -128 ≤ d2 ∧ d2 ≤ 127 := by
  have := digits_bounds T h1 h2
  simp only [digits] at this ⊢
  omega

/-! ## The magic constant `1.5 * 2^23` -/

/-- `C = 1.5 * 2^23 = 12582912`. -/
local notation "C" => (12582912 : Int)

/-- Adding the even integer `C` commutes with `rn`: `rn(t + C) = rn(t) + C`. In `[2^23, 2^24)`
float32 values are the integers, so float32 round to nearest even of `t + C` is `rn(t + C)`. -/
theorem rn_add_C (T : Int) : rn (T + C * S) = rn T + C := by
  unfold rn; split <;> split <;> omega

/-- Decoding float32 bit patterns of the form `0x4B400000 + i` (`|i| < 2^22`): sign 0, exponent
field 150, fraction `2^22 + i`, so the value is `(2^23 + fraction) * 2^(150 - 150) = C + i`. -/
theorem magic_decode (i : Int) (h1 : -(2 ^ 22) ≤ i) (h2 : i < 2 ^ 22) :
    let b := 0x4B400000 + i
    0 ≤ b ∧ b / 2 ^ 31 = 0 ∧ b / 2 ^ 23 % 256 = 150 ∧ 2 ^ 23 + b % 2 ^ 23 = C + i := by
  omega

/-! ## Integer sums of one K block -/

/-- The three digits of one operand value. -/
structure Dig where
  d0 : Int
  d1 : Int
  d2 : Int

def Dig.ok (x : Dig) : Prop :=
  x.d0.natAbs ≤ 127 ∧ x.d1.natAbs ≤ 64 ∧ x.d2.natAbs ≤ 64

def Dig.get (x : Dig) : Nat → Int
  | 0 => x.d0
  | 1 => x.d1
  | _ => x.d2

/-- Digit index pairs `(p, q)` of weight `w = p + q` that the kernel forms (`w <= 2`). -/
def pairs : Nat → List (Nat × Nat)
  | 0 => [(0, 0)]
  | 1 => [(0, 1), (1, 0)]
  | _ => [(0, 2), (1, 1), (2, 0)]

/-- All digit products of weight `w` of one K block (columns `k`, pairs `(a_k, b_k)`). -/
def products (w : Nat) (cols : List (Dig × Dig)) : List Int :=
  cols.flatMap (fun c => (pairs w).map (fun pq => c.1.get pq.1 * c.2.get pq.2))

/-- Per column bound of the absolute products of weight `w`. -/
def colBound : Nat → Nat
  | 0 => 127 * 127
  | 1 => 2 * 127 * 64
  | _ => 2 * 127 * 64 + 64 * 64

theorem get_le (x : Dig) (hx : x.ok) (p : Nat) :
    (x.get p).natAbs ≤ (if p = 0 then 127 else 64) := by
  obtain ⟨h0, h1, h2⟩ := hx
  match p with
  | 0 => simpa [Dig.get] using h0
  | 1 => simpa [Dig.get] using h1
  | n + 2 => simpa [Dig.get] using h2

theorem prod_le (a b : Dig) (ha : a.ok) (hb : b.ok) (p q : Nat) :
    (a.get p * b.get q).natAbs ≤ (if p = 0 then 127 else 64) * (if q = 0 then 127 else 64) := by
  rw [Int.natAbs_mul]
  exact Nat.mul_le_mul (get_le a ha p) (get_le b hb q)

theorem col_le (w : Nat) (a b : Dig) (ha : a.ok) (hb : b.ok) :
    sumBy (fun z => (z.natAbs : Int)) ((pairs w).map (fun pq => a.get pq.1 * b.get pq.2)) ≤
      (colBound w : Int) := by
  have h := prod_le a b ha hb
  match w with
  | 0 =>
    have := h 0 0
    simp [pairs, colBound, sumBy] at *; omega
  | 1 =>
    have := h 0 1; have := h 1 0
    simp [pairs, colBound, sumBy] at *; omega
  | n + 2 =>
    have := h 0 2; have := h 1 1; have := h 2 0
    simp [pairs, colBound, sumBy] at *; omega

theorem products_abs_le (w : Nat) :
    ∀ cols : List (Dig × Dig), (∀ c ∈ cols, c.1.ok ∧ c.2.ok) →
      sumBy (fun z => (z.natAbs : Int)) (products w cols) ≤ cols.length * (colBound w : Int) := by
  intro cols
  induction cols with
  | nil => intro _; simp [products]
  | cons c cols ih =>
    intro h
    have hc := h c (by simp)
    have := col_le w c.1 c.2 hc.1 hc.2
    have := ih (fun x hx => h x (by simp [hx]))
    simp only [products, List.flatMap_cons, sumBy_append] at *
    simp only [List.length_cons]
    rw [Int.natCast_add, Int.add_mul]
    omega

/-- **T3, sums.** Every weight's absolute sum over a K block of at most 128 columns is at most
`128 * 20352 = 2,605,056 < 2^22`. -/
theorem block_sum_bound (w : Nat) (cols : List (Dig × Dig)) (hk : cols.length ≤ 128)
    (h : ∀ c ∈ cols, c.1.ok ∧ c.2.ok) :
    sumBy (fun z => (z.natAbs : Int)) (products w cols) ≤ 2605056 ∧ (2605056 : Int) < 2 ^ 22 := by
  have := products_abs_le w cols h
  have hb : (colBound w : Int) ≤ 20352 := by
    match w with
    | 0 => simp [colBound]
    | 1 => simp [colBound]
    | n + 2 => simp [colBound]
  have hb0 : (0 : Int) ≤ colBound w := by omega
  refine ⟨?_, by decide⟩
  have : (cols.length : Int) * colBound w ≤ 128 * 20352 :=
    Int.mul_le_mul (by omega) hb hb0 (by omega)
  omega

end Truestar.Digits
