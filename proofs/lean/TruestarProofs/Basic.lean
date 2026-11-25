/-!
Small list lemmas shared by the proofs (Lean core only, no mathlib).
-/
namespace Truestar

/-- Sum of `f` over a list, as an integer. -/
def sumBy {α : Type} (f : α → Int) (l : List α) : Int := (l.map f).sum

@[simp] theorem sumBy_nil {α : Type} (f : α → Int) : sumBy f [] = 0 := rfl

@[simp] theorem sumBy_cons {α : Type} (f : α → Int) (a : α) (l : List α) :
    sumBy f (a :: l) = f a + sumBy f l := by
  simp [sumBy, List.sum_cons]

theorem sumBy_append {α : Type} (f : α → Int) (l₁ l₂ : List α) :
    sumBy f (l₁ ++ l₂) = sumBy f l₁ + sumBy f l₂ := by
  induction l₁ with
  | nil => simp
  | cons a l ih => simp [ih]; omega

theorem sumBy_flatten_cons {α : Type} (f : α → Int) (g : List α) (gs : List (List α)) :
    sumBy f (g :: gs).flatten = sumBy f g + sumBy f gs.flatten := by
  simp [List.flatten_cons, sumBy_append]

/-- A sum does not depend on the order of its terms (integer addition). -/
theorem sumBy_perm {α : Type} (f : α → Int) {l₁ l₂ : List α} (h : l₁.Perm l₂) :
    sumBy f l₁ = sumBy f l₂ := by
  induction h with
  | nil => rfl
  | cons a _ ih => simp [ih]
  | swap a b l => simp; omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem sumBy_congr {α : Type} (f g : α → Int) (l : List α) (h : ∀ a ∈ l, f a = g a) :
    sumBy f l = sumBy g l := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [sumBy_cons]
    rw [h a (by simp), ih (fun x hx => h x (by simp [hx]))]

/-- |sum| <= sum of |.| (two sided form). -/
theorem sumBy_abs_le {α : Type} (f : α → Int) (l : List α) :
    -sumBy (fun a => ((f a).natAbs : Int)) l ≤ sumBy f l ∧
      sumBy f l ≤ sumBy (fun a => ((f a).natAbs : Int)) l := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [sumBy_cons]; omega

theorem sumBy_nonneg {α : Type} (f : α → Int) (l : List α) (h : ∀ a ∈ l, 0 ≤ f a) :
    0 ≤ sumBy f l := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [sumBy_cons]
    have := h a (by simp)
    have := ih (fun x hx => h x (by simp [hx]))
    omega

theorem sumBy_mul_right {α : Type} (f : α → Int) (l : List α) (k : Int) :
    sumBy (fun a => f a * k) l = sumBy f l * k := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [sumBy_cons, ih, Int.add_mul]

theorem le_sumBy_of_mem {α : Type} (f : α → Int) (l : List α) (h : ∀ a ∈ l, 0 ≤ f a)
    (a : α) (ha : a ∈ l) : f a ≤ sumBy f l := by
  induction l with
  | nil => simp at ha
  | cons b l ih =>
    simp only [sumBy_cons]
    have hb := h b (by simp)
    have hrest := sumBy_nonneg f l (fun x hx => h x (by simp [hx]))
    rcases List.mem_cons.mp ha with rfl | ha'
    · omega
    · have := ih (fun x hx => h x (by simp [hx])) ha'
      omega

end Truestar
