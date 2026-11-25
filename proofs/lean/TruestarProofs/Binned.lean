import TruestarProofs.Determinism

/-!
# T6: order free (binned) sum, any tree gives the same bits

`src/truestar/kernel/ordersum.py`: every term becomes an integer `q` on one grid chosen from the
row's maximum exponent (a maximum, itself order free), `|q| < 2^39`, at most `2^24 - 1` terms; the
integers are added in int64 and the total is rounded to float32 once.

Proved here:
* for any operation that is associative and commutative with an identity, the value of any
  reduction tree depends only on the multiset of its leaves (`tree_perm`);
* int64 wrapping addition (`BitVec 64`) is such an operation, so any tree of int64 additions of
  the `q`s equals the int64 encoding of the exact integer sum (`osum_tree`);
* with `n <= 2^24 - 1` terms below `2^39` the exact sum is inside the int64 range, so the int64
  total read back as a signed integer is the exact sum (`osum_exact`); the single final rounding
  then sees the same integer whatever the order.
-/
namespace Truestar.Binned

open Truestar.Det

/-- Leaves of a tree, left to right. -/
def leaves : Tree → List Nat
  | .leaf i => [i]
  | .node l r => leaves l ++ leaves r

section
variable {α : Type} (op : α → α → α) (z : α)
  (assoc : ∀ a b c, op (op a b) c = op a (op b c))
  (comm : ∀ a b, op a b = op b a)
  (idl : ∀ a, op z a = a)

def fold (l : List α) : α := l.foldr op z

include assoc idl in
theorem fold_append (l₁ l₂ : List α) : fold op z (l₁ ++ l₂) = op (fold op z l₁) (fold op z l₂) := by
  induction l₁ with
  | nil => simp [fold, idl]
  | cons a l ih => simp only [fold, List.cons_append, List.foldr_cons] at *; rw [ih, assoc]

include assoc comm idl in
theorem eval_eq_fold (x : Nat → α) (t : Tree) : eval op x t = fold op z ((leaves t).map x) := by
  induction t with
  | leaf i =>
    simp only [eval, leaves, List.map, fold, List.foldr_cons, List.foldr_nil]
    rw [comm, idl]
  | node l r ihl ihr =>
    simp only [eval, leaves, List.map_append]
    rw [fold_append op z assoc idl, ihl, ihr]

include assoc comm in
theorem fold_perm {l₁ l₂ : List α} (h : l₁.Perm l₂) : fold op z l₁ = fold op z l₂ := by
  induction h with
  | nil => rfl
  | cons a _ ih => simp only [fold, List.foldr_cons] at *; rw [ih]
  | swap a b l =>
    simp only [fold, List.foldr_cons]
    rw [← assoc, comm b a, assoc]
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

include assoc comm idl in
/-- Any two trees over the same leaves (in any order) give the same value. -/
theorem tree_perm (x : Nat → α) (t₁ t₂ : Tree) (h : (leaves t₁).Perm (leaves t₂)) :
    eval op x t₁ = eval op x t₂ := by
  rw [eval_eq_fold op z assoc comm idl, eval_eq_fold op z assoc comm idl]
  exact fold_perm op z assoc comm (h.map x)
end

/-- int64 wrapping addition is associative, commutative, with identity 0. -/
theorem osum_tree (q : Nat → Int) (t : Tree) :
    eval (· + ·) (fun i => BitVec.ofInt 64 (q i)) t =
      BitVec.ofInt 64 (((leaves t).map q).sum) := by
  induction t with
  | leaf i => simp [eval, leaves]
  | node l r ihl ihr =>
    simp only [eval, leaves, List.map_append, List.sum_append, ihl, ihr, BitVec.ofInt_add]

theorem osum_order_free (q : Nat → Int) (t₁ t₂ : Tree) (h : (leaves t₁).Perm (leaves t₂)) :
    eval (· + ·) (fun i => BitVec.ofInt 64 (q i)) t₁ =
      eval (· + ·) (fun i => BitVec.ofInt 64 (q i)) t₂ :=
  tree_perm (· + ·) 0 (fun a b c => BitVec.add_assoc a b c) (fun a b => BitVec.add_comm a b)
    (fun a => BitVec.zero_add a) _ t₁ t₂ h

theorem sum_bound (l : List Int) (h : ∀ v ∈ l, -(2 ^ 39) < v ∧ v < 2 ^ 39) :
    -(l.length * (2 ^ 39 : Int)) ≤ l.sum ∧ l.sum ≤ l.length * (2 ^ 39 : Int) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    have ha := h a (by simp)
    have := ih (fun v hv => h v (by simp [hv]))
    simp only [List.sum_cons, List.length_cons, Int.natCast_add, Int.add_mul]
    omega

/-- With at most `2^24 - 1` terms below `2^39` the int64 total read as a signed integer is the
exact integer sum, for every tree. -/
theorem osum_exact (q : Nat → Int) (t : Tree) (hn : (leaves t).length ≤ 2 ^ 24 - 1)
    (hq : ∀ i ∈ leaves t, -(2 ^ 39) < q i ∧ q i < 2 ^ 39) :
    (eval (· + ·) (fun i => BitVec.ofInt 64 (q i)) t).toInt = ((leaves t).map q).sum := by
  rw [osum_tree, BitVec.toInt_ofInt]
  have hb := sum_bound ((leaves t).map q) (by
    intro v hv; obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hv; exact hq i hi)
  simp only [List.length_map] at hb
  have hlen : ((leaves t).length : Int) * 2 ^ 39 ≤ (2 ^ 24 - 1) * 2 ^ 39 :=
    Int.mul_le_mul_of_nonneg_right (by omega) (by decide)
  apply Int.bmod_eq_of_le <;> omega

end Truestar.Binned
