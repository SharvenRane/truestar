import Mathlib

/-!
# T7: the online softmax rescaling identity

Flash style attention keeps, per query, a running reference `m` (the running maximum), a running
denominator `s = sum_j exp(x_j - m)` and a running weighted sum `o = sum_j exp(x_j - m) v_j` over
the keys seen so far. When the reference moves to `m'`, both are rescaled by `exp(m - m')`; a new
key `(x, v)` adds `exp(x - m')` and `exp(x - m') v`; two partial states (two key blocks) merge by
rescaling both to a common reference and adding.

Proved in real arithmetic (no rounding):
* `rescale`, `push`, `merge` keep the invariant "the state represents the list of keys seen";
* `online_softmax`: for any sequence of steps and block merges, and any choice of the reference
  values (the running maximum is only one choice, made to avoid overflow), the final `o / s`
  equals `sum_j softmax(x)_j v_j` exactly.

The rounding error bound in terms of the key block width (the second half of T7 in
LUMEN_MATH_FOUNDATIONS.md section 11) is not formalized here.
-/
namespace Truestar.Softmax

open Real

noncomputable section

/-- Keys seen: pairs (score `x`, value `v`). -/
abbrev Keys := List (ℝ × ℝ)

structure St where
  m : ℝ
  s : ℝ
  o : ℝ

/-- `st` represents the keys `ks`. -/
def Rep (st : St) (ks : Keys) : Prop :=
  st.s = (ks.map fun k => exp (k.1 - st.m)).sum ∧
    st.o = (ks.map fun k => exp (k.1 - st.m) * k.2).sum

def rescale (st : St) (m' : ℝ) : St := ⟨m', st.s * exp (st.m - m'), st.o * exp (st.m - m')⟩

def push (st : St) (k : ℝ × ℝ) (m' : ℝ) : St :=
  let r := rescale st m'
  ⟨m', r.s + exp (k.1 - m'), r.o + exp (k.1 - m') * k.2⟩

def merge (a b : St) (m' : ℝ) : St :=
  let ra := rescale a m'
  let rb := rescale b m'
  ⟨m', ra.s + rb.s, ra.o + rb.o⟩

def empty (m : ℝ) : St := ⟨m, 0, 0⟩

theorem rep_empty (m : ℝ) : Rep (empty m) [] := by simp [Rep, empty]

theorem sum_rescale (ks : Keys) (m m' : ℝ) (f : ℝ × ℝ → ℝ) :
    (ks.map fun k => exp (k.1 - m) * f k).sum * exp (m - m') =
      (ks.map fun k => exp (k.1 - m') * f k).sum := by
  induction ks with
  | nil => simp
  | cons k ks ih =>
    simp only [List.map_cons, List.sum_cons, add_mul, ih]
    congr 1
    have : exp (k.1 - m') = exp (k.1 - m) * exp (m - m') := by rw [← exp_add]; ring_nf
    rw [this]; ring

theorem rep_rescale (st : St) (ks : Keys) (h : Rep st ks) (m' : ℝ) : Rep (rescale st m') ks := by
  obtain ⟨hs, ho⟩ := h
  constructor
  · have := sum_rescale ks st.m m' (fun _ => 1)
    simp only [mul_one] at this
    simp only [rescale, hs]; exact this
  · simp only [rescale, ho]; exact sum_rescale ks st.m m' (fun k => k.2)

theorem rep_push (st : St) (ks : Keys) (h : Rep st ks) (k : ℝ × ℝ) (m' : ℝ) :
    Rep (push st k m') (ks ++ [k]) := by
  obtain ⟨hs, ho⟩ := rep_rescale st ks h m'
  constructor
  · simp only [push, List.map_append, List.sum_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, add_zero]
    rw [hs]; rfl
  · simp only [push, List.map_append, List.sum_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, add_zero]
    rw [ho]; rfl

theorem rep_merge (a b : St) (ka kb : Keys) (ha : Rep a ka) (hb : Rep b kb) (m' : ℝ) :
    Rep (merge a b m') (ka ++ kb) := by
  obtain ⟨has, hao⟩ := rep_rescale a ka ha m'
  obtain ⟨hbs, hbo⟩ := rep_rescale b kb hb m'
  constructor
  · simp only [merge, List.map_append, List.sum_append]; rw [has, hbs]; rfl
  · simp only [merge, List.map_append, List.sum_append]; rw [hao, hbo]; rfl

/-- Softmax weighted sum of the values. -/
def attn (ks : Keys) : ℝ :=
  (ks.map fun k => exp k.1 * k.2).sum / (ks.map fun k => exp k.1).sum

/-- **T7.** A state that represents a nonempty key list gives `o / s = attn ks`, whatever its
reference `m`. -/
theorem online_softmax (st : St) (ks : Keys) (hne : ks ≠ []) (h : Rep st ks) :
    st.o / st.s = attn ks := by
  obtain ⟨hs, ho⟩ := h
  have hpos : 0 < (ks.map fun k => exp k.1).sum := by
    obtain ⟨k, ks', rfl⟩ := List.exists_cons_of_ne_nil hne
    simp only [List.map_cons, List.sum_cons]
    exact add_pos_of_pos_of_nonneg (exp_pos _)
      (List.sum_nonneg (by intro y hy; obtain ⟨z, _, rfl⟩ := List.mem_map.mp hy; exact (exp_pos _).le))
  have e1 := sum_rescale ks 0 st.m (fun _ => 1)
  have e2 := sum_rescale ks 0 st.m (fun k => k.2)
  simp only [sub_zero, mul_one] at e1 e2
  rw [hs, ho, ← e1, ← e2, attn, mul_div_mul_right _ _ (exp_pos _).ne']

end

end Truestar.Softmax
