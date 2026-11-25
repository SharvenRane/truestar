import Mathlib

/-!
# T4: error bound of a reduction tree

A sum of inputs `x_i` evaluated over a binary tree `T`, every addition rounded with a relative
error `delta` of magnitude at most `u` (the standard model `fl(a + b) = (a + b)(1 + delta)`; for
float32 round to nearest `u = 2^-24`, valid without underflow and overflow). The rounding at each
node is arbitrary (adversarial): it is any function of the node's position.

**T4.** `|fl_T(x) - sum x| <= ((1 + u)^depth(T) - 1) * sum |x|`.

`(1 + u)^d - 1 <= gamma_d = d u / (1 - d u)` (Higham, Lemma 3.1), so this is the usual
`gamma_depth` bound in its sharper intermediate form.
-/
namespace Truestar.TreeBound

inductive Tree where
  | leaf : ℕ → Tree
  | node : Tree → Tree → Tree

def depth : Tree → ℕ
  | .leaf _ => 0
  | .node l r => max (depth l) (depth r) + 1

def exact (x : ℕ → ℝ) : Tree → ℝ
  | .leaf i => x i
  | .node l r => exact x l + exact x r

def absum (x : ℕ → ℝ) : Tree → ℝ
  | .leaf i => |x i|
  | .node l r => absum x l + absum x r

/-- Rounded evaluation: the addition at position `p` (path from the root) has relative error
`δ p`. -/
def fl (x : ℕ → ℝ) (δ : List Bool → ℝ) (p : List Bool) : Tree → ℝ
  | .leaf i => x i
  | .node l r => (fl x δ (p ++ [false]) l + fl x δ (p ++ [true]) r) * (1 + δ p)

theorem absum_nonneg (x : ℕ → ℝ) : ∀ t, 0 ≤ absum x t
  | .leaf _ => abs_nonneg _
  | .node l r => add_nonneg (absum_nonneg x l) (absum_nonneg x r)

theorem abs_exact_le (x : ℕ → ℝ) : ∀ t, |exact x t| ≤ absum x t
  | .leaf _ => le_refl _
  | .node l r => (abs_add_le _ _).trans (add_le_add (abs_exact_le x l) (abs_exact_le x r))

theorem tree_bound (x : ℕ → ℝ) (u : ℝ) (hu : 0 ≤ u) (δ : List Bool → ℝ) (hδ : ∀ p, |δ p| ≤ u) :
    ∀ (t : Tree) (p : List Bool),
      |fl x δ p t - exact x t| ≤ ((1 + u) ^ depth t - 1) * absum x t := by
  intro t
  induction t with
  | leaf i => intro p; simp [fl, exact, depth]
  | node l r ihl ihr =>
    intro p
    simp only [fl, exact, depth, absum]
    set d := max (depth l) (depth r)
    set k := (1 + u) ^ d
    have h1 : (1 : ℝ) ≤ 1 + u := by linarith
    have hk : 1 ≤ k := one_le_pow₀ h1
    have hl : |fl x δ (p ++ [false]) l - exact x l| ≤ (k - 1) * absum x l := by
      refine (ihl _).trans (mul_le_mul_of_nonneg_right ?_ (absum_nonneg x l))
      have := pow_le_pow_right₀ h1 (le_max_left (depth l) (depth r)); linarith
    have hr : |fl x δ (p ++ [true]) r - exact x r| ≤ (k - 1) * absum x r := by
      refine (ihr _).trans (mul_le_mul_of_nonneg_right ?_ (absum_nonneg x r))
      have := pow_le_pow_right₀ h1 (le_max_right (depth l) (depth r)); linarith
    have hA := abs_exact_le x l
    have hB := abs_exact_le x r
    have hd := hδ p
    set a := fl x δ (p ++ [false]) l
    set b := fl x δ (p ++ [true]) r
    set A := exact x l
    set B := exact x r
    set S := absum x l + absum x r
    have key : (a + b) * (1 + δ p) - (A + B) = ((a - A) + (b - B)) * (1 + δ p) + (A + B) * δ p := by
      ring
    have e1 : |(a - A) + (b - B)| ≤ (k - 1) * S := by
      calc |(a - A) + (b - B)| ≤ |a - A| + |b - B| := abs_add_le _ _
        _ ≤ (k - 1) * absum x l + (k - 1) * absum x r := add_le_add hl hr
        _ = (k - 1) * S := by ring
    have e2 : |1 + δ p| ≤ 1 + u := by
      calc |1 + δ p| ≤ |(1 : ℝ)| + |δ p| := abs_add_le _ _
        _ ≤ 1 + u := by rw [abs_one]; linarith
    have e3 : |A + B| ≤ S := (abs_add_le _ _).trans (add_le_add hA hB)
    have hS : 0 ≤ S := add_nonneg (absum_nonneg x l) (absum_nonneg x r)
    rw [key, pow_succ]
    calc |((a - A) + (b - B)) * (1 + δ p) + (A + B) * δ p|
        ≤ |(a - A) + (b - B)| * |1 + δ p| + |A + B| * |δ p| := by
          rw [← abs_mul, ← abs_mul]; exact abs_add_le _ _
      _ ≤ ((k - 1) * S) * (1 + u) + S * u := by
          gcongr
      _ = (k * (1 + u) - 1) * S := by ring

/-- The same bound for any choice of the rounding errors: the tree, not the rounding, fixes it. -/
theorem tree_bound_root (x : ℕ → ℝ) (u : ℝ) (hu : 0 ≤ u) (δ : List Bool → ℝ)
    (hδ : ∀ p, |δ p| ≤ u) (t : Tree) :
    |fl x δ [] t - exact x t| ≤ ((1 + u) ^ depth t - 1) * absum x t :=
  tree_bound x u hu δ hδ t []

/-- `(1 + u)^d - 1 <= d u / (1 - d u)` for `d u < 1`: the bound is at most `gamma_depth`. -/
theorem pow_mul_one_sub_le (u : ℝ) (hu : 0 ≤ u) : ∀ n : ℕ, (1 + u) ^ n * (1 - n * u) ≤ 1
  | 0 => by simp
  | n + 1 => by
    have ih := pow_mul_one_sub_le u hu n
    have hP : 0 ≤ (1 + u) ^ n := pow_nonneg (by linarith) n
    have e : (1 + u) ^ (n + 1) * (1 - ((n + 1 : ℕ) : ℝ) * u) =
        (1 + u) ^ n * (1 - n * u) - (1 + u) ^ n * ((n + 1) * u ^ 2) := by
      push_cast; ring
    have h2 : 0 ≤ (1 + u) ^ n * ((n + 1) * u ^ 2) :=
      mul_nonneg hP (mul_nonneg (by positivity) (sq_nonneg u))
    rw [e]; linarith

theorem pow_sub_one_le_gamma (u : ℝ) (hu : 0 ≤ u) (d : ℕ) (hdu : d * u < 1) :
    (1 + u) ^ d - 1 ≤ d * u / (1 - d * u) := by
  have hpos : 0 < 1 - d * u := by linarith
  rw [le_div_iff₀ hpos]
  have := pow_mul_one_sub_le u hu d
  nlinarith

end Truestar.TreeBound
