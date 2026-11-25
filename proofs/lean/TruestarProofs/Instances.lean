import TruestarProofs.Constants
import TruestarProofs.TCWindow
import TruestarProofs.Digits

/-!
# The theorems at Truestar's constants

`Constants.lean` is generated from `src/truestar/kernel/tcexact.py`. Every side condition below
is checked by `decide` on those values, so `lake build` fails if a constant is changed in a way
the proofs do not cover:

* `TC_FRAC_BITS_MIN` below 23 (the proof needs a grid no coarser than one integer at `E = 23`);
  and, as a separate guard on the measured hardware value, below 24 (`frac_bits_measured`);
* `ACC_START` (C) such that `C - 2,605,056 < 2^23` or `C + 2,605,056 >= 2^24`;
* `MAX_KB` above 128, `PORTABLE_BK` above `MAX_KB`, digits wider than 127 / 64, `DIGIT_BITS` not 7,
  `DIGITS` not 3, or `SUM_BITS` such that the block sums do not fit.
-/
namespace Truestar.Inst

open Truestar Truestar.TC Truestar.Digits Truestar.Const

/-- The proof's requirement on the window. -/
theorem frac_bits_ok : 23 ≤ TC_FRAC_BITS_MIN := by decide

/-- Guard on the measured value (sm_80, sm_86, sm_89 keep 24 fraction bits; the xchip probe). This
is not needed by the proof, which holds from 23; it pins the constant to the measured hardware. -/
theorem frac_bits_measured : 24 ≤ TC_FRAC_BITS_MIN := by decide

theorem digit_consts_ok : TOP_MAX = 127 ∧ LOW_MAX = 64 ∧ DIGIT_BITS = 7 ∧ DIGITS = 3 ∧
    2 ^ DIGIT_BITS / 2 = LOW_MAX ∧ TOP_MAX ≤ 127 := by decide

theorem block_consts_ok : MAX_KB ≤ 128 ∧ PORTABLE_BK ≤ MAX_KB := by decide

/-- The accumulator window: `C - W >= 2^23` and `C + W < 2^24` with `W = 2,605,056`. -/
theorem acc_window_ok : 2 ^ 23 ≤ ACC_START - 2605056 ∧ ACC_START + 2605056 < 2 ^ 24 := by decide

/-- `ACC_START` is the float32 rounding constant 1.5 * 2^23 of the digit code too. -/
theorem acc_is_rn_constant : ACC_START = 12582912 := by decide

theorem sum_bits_ok : (2605056 : Int) < 2 ^ SUM_BITS ∧ SUM_BITS ≤ 22 := by decide

theorem sumBy_map {α β : Type} (f : β → Int) (g : α → β) (l : List α) :
    sumBy f (l.map g) = sumBy (fun a => f (g a)) l := by
  simp [sumBy, List.map_map, Function.comp_def]

/-- **The fp16 engine of tcexact, exact.** For a K block of at most `MAX_KB` columns whose digits
satisfy the T3 ranges, and any weight `w`: on any matrix unit whose window has at least
`TC_FRAC_BITS_MIN` fraction bits (with any group size, cut mode and exponent floor up to `2^23`),
whatever grouping and order the products are fed in, the chain started at `ACC_START` returns
`ACC_START + I_w` exactly, where `I_w` is the exact integer sum of the digit products. -/
theorem fp16_engine_exact (u : MatrixUnit) (round : Int → Int) (cexp : Int → Int) (L : Nat)
    (hF : TC_FRAC_BITS_MIN ≤ u.F) (hfl : ∀ f, u.floor = some f → f ≤ 23 + L)
    (hround : ∀ n : Int, n.natAbs < 2 ^ 24 → round (n * 2 ^ L) = n * 2 ^ L)
    (hc : ∀ c : Int, c ≠ 0 → cexp c ≤ (Nat.log2 c.natAbs : Int))
    (w : Nat) (cols : List (Dig × Dig)) (hk : cols.length ≤ MAX_KB)
    (hdig : ∀ c ∈ cols, c.1.ok ∧ c.2.ok)
    (ps : List (Int × Int)) (hps : ps.map Prod.fst = products w cols)
    (hexp : ∀ x ∈ ps, x.1 ≠ 0 → x.2 ≤ (Nat.log2 (x.1 * 2 ^ L).natAbs : Int))
    (gs : List (List (Int × Int))) (hperm : gs.flatten.Perm ps) :
    chain u round cexp (ACC_START * 2 ^ L) (gs.map (List.map (prodTerm L))) =
      (ACC_START + (products w cols).sum) * 2 ^ L := by
  have hF23 : 23 ≤ u.F := Nat.le_trans frac_bits_ok hF
  have hk' : cols.length ≤ 128 := Nat.le_trans hk block_consts_ok.1
  have hb := (block_sum_bound w cols hk' hdig).1
  have habs : absSum ps = sumBy (fun z => (z.natAbs : Int)) (products w cols) := by
    rw [← hps, sumBy_map]; rfl
  have hsum : sumBy Prod.fst ps = (products w cols).sum := by
    rw [← hps]; simp [sumBy]
  have hwin := acc_window_ok
  rw [chain_exact_perm u round cexp L hF23 hfl hround hc ps gs hperm hexp ACC_START
    (by omega) (by omega), hsum]

end Truestar.Inst
