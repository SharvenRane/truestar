import TruestarProofs.Basic

/-!
# T1: the exact tensor core window

## The hardware model (what is assumed)

One fused group of a tensor core instruction (docs/TENSOR_CORE_MODELS.md, the model of
`src/truestar/devsim/tensorcore.py`, after Khattak and Mikaitis) takes a list of terms: the exact,
unnormalized products of one group and the accumulator `c`. It

1. reads an exponent `e` for every nonzero term (for a product the sum of the two operand
   exponents, for `c` its normalized exponent) and takes the largest, `E`; a unit with an exponent
   floor raises `E` to at least the floor;
2. cuts every term to the grid `2^(E - F)` (`F` fraction bits): toward zero (`rz`) or toward minus
   infinity (`rd`, two's complement cut);
3. adds the cut terms exactly;
4. rounds the sum once to the accumulator format.

A dot product longer than one group chains groups: the rounded result of one group is the `c` of
the next.

Values are represented exactly as integers in units of `2^-L` for a fixed `L` (every value the
model ever sees is a dyadic rational, so a large enough `L` loses nothing); exponents are in the
same units (value exponent plus `L`). Every step above is then integer arithmetic, defined below
by `group` and `chain`. The final rounding is any function `round`; the only property used is
that it returns every integer of magnitude below `2^24` unchanged (true of truncation and of
round to nearest even into float32, since such integers are float32 values). The only property
used of the exponents a unit reads is that the exponent of a nonzero term is never above the
exponent of its true magnitude (`e <= log2 |v|`): true for normalized values and for products of
normal operands, which is what the digits are.

`TcCheck.lean` runs the same `group` / `chain` functions on probe vectors exported from the Python
simulator and compares the bits.
-/
namespace Truestar.TC

/-- One term of a fused group: value `v` and the exponent `e` the unit reads, both in units of
`2^-L`. -/
structure Term where
  v : Int
  e : Int
deriving Repr

inductive Cut where
  | rz
  | rd
deriving Repr, DecidableEq

/-- Cut `v` onto the grid `2^s` (in units). A grid finer than one unit drops nothing. -/
def cut (k : Cut) (v : Int) (s : Int) : Int :=
  if s ≤ 0 then v else
    match k with
    | .rz => v.tdiv (2 ^ s.toNat) * 2 ^ s.toNat
    | .rd => v / 2 ^ s.toNat * 2 ^ s.toNat

/-- The largest exponent among the nonzero terms (none when every term is zero). -/
def topExp : List Term → Option Int
  | [] => none
  | t :: ts =>
    match topExp ts with
    | none => if t.v = 0 then none else some t.e
    | some a => if t.v = 0 then some a else some (max t.e a)

def withFloor (floor : Option Int) (E : Int) : Int :=
  match floor with
  | none => E
  | some f => max E f

/-- Parameters of one matrix unit data path: fraction bits of the alignment grid, how bits below
it are cut, and the exponent floor (in units) if the unit has one. -/
structure MatrixUnit where
  F : Nat
  cutMode : Cut
  floor : Option Int

/-- One fused group: align to the largest exponent, cut, add exactly, round once. -/
def group (u : MatrixUnit) (round : Int → Int) (ts : List Term) : Int :=
  match topExp ts with
  | none => 0
  | some E => round (sumBy (fun t => cut u.cutMode t.v (withFloor u.floor E - u.F)) ts)

/-- A chain of groups along K, in order; `cexp` is the exponent the unit reads from `c`. -/
def chain (u : MatrixUnit) (round : Int → Int) (cexp : Int → Int) (c : Int) :
    List (List Term) → Int
  | [] => c
  | g :: gs => chain u round cexp (group u round (⟨c, cexp c⟩ :: g)) gs

/-! ## Lemmas -/

theorem topExp_le (B : Int) :
    ∀ ts : List Term, (∀ t ∈ ts, t.v ≠ 0 → t.e ≤ B) → ∀ E, topExp ts = some E → E ≤ B := by
  intro ts
  induction ts with
  | nil => intro _ E h; simp [topExp] at h
  | cons t ts ih =>
    intro hb E h
    have hrest : ∀ x ∈ ts, x.v ≠ 0 → x.e ≤ B := fun x hx => hb x (by simp [hx])
    have ht : t.v ≠ 0 → t.e ≤ B := hb t (by simp)
    simp only [topExp] at h
    split at h
    · split at h
      · simp at h
      · simp at h; subst h; exact ht (by assumption)
    · rename_i a ha
      have hab := ih hrest a ha
      split at h
      · simp at h; subst h; exact hab
      · simp at h; subst h
        have := ht (by assumption)
        simp only [Int.max_def]; split <;> omega

theorem topExp_none : ∀ ts : List Term, topExp ts = none → ∀ t ∈ ts, t.v = 0 := by
  intro ts
  induction ts with
  | nil => simp
  | cons t ts ih =>
    intro h x hx
    simp only [topExp] at h
    split at h
    · rename_i hn
      split at h
      · rcases List.mem_cons.mp hx with rfl | hx'
        · assumption
        · exact ih hn x hx'
      · simp at h
    · split at h <;> simp at h

theorem pow_dvd_pow_int (a L : Nat) (h : a ≤ L) : (2 : Int) ^ a ∣ (2 : Int) ^ L :=
  ⟨2 ^ (L - a), by rw [← Int.pow_add]; congr 1; omega⟩

/-- A grid no coarser than one integer (in units of `2^-L`) leaves a multiple of `2^L` unchanged. -/
theorem cut_id (k : Cut) (v s : Int) (L : Nat) (hs : s ≤ L) (hv : (2 : Int) ^ L ∣ v) :
    cut k v s = v := by
  unfold cut
  split
  · rfl
  · have hd : (2 : Int) ^ s.toNat ∣ v := Int.dvd_trans (pow_dvd_pow_int _ _ (by omega)) hv
    cases k
    · exact Int.tdiv_mul_cancel hd
    · exact Int.ediv_mul_cancel hd

/-- A term below `2^24` integers has an exponent of at most `23` (in units: `23 + L`). -/
theorem exp_le (L : Nat) (t : Term) (q : Int) (hv : t.v = q * 2 ^ L) (hq : q.natAbs < 2 ^ 24)
    (hne : t.v ≠ 0) (he : t.e ≤ (Nat.log2 t.v.natAbs : Int)) : t.e ≤ 23 + L := by
  have habs : t.v.natAbs = q.natAbs * 2 ^ L := by
    rw [hv, Int.natAbs_mul, Int.natAbs_pow]; rfl
  have hnz : t.v.natAbs ≠ 0 := by omega
  have hlt : t.v.natAbs < 2 ^ (24 + L) := by
    rw [habs, Nat.pow_add]
    exact Nat.mul_lt_mul_of_pos_right hq (Nat.two_pow_pos L)
  have := (Nat.log2_lt hnz).mpr hlt
  omega

/-- One group is exact when every term is an integer below `2^24`, the exact sum is an integer
below `2^24`, the grid has at least 23 fraction bits and the floor is not above `2^23`. -/
theorem group_exact (u : MatrixUnit) (round : Int → Int) (L : Nat)
    (hF : 23 ≤ u.F) (hfl : ∀ f, u.floor = some f → f ≤ 23 + L)
    (hround : ∀ n : Int, n.natAbs < 2 ^ 24 → round (n * 2 ^ L) = n * 2 ^ L)
    (ts : List Term)
    (hts : ∀ t ∈ ts, ∃ q : Int, t.v = q * 2 ^ L ∧ q.natAbs < 2 ^ 24 ∧
      (t.v ≠ 0 → t.e ≤ (Nat.log2 t.v.natAbs : Int)))
    (n : Int) (hsum : sumBy Term.v ts = n * 2 ^ L) (hn : n.natAbs < 2 ^ 24) :
    group u round ts = n * 2 ^ L := by
  unfold group
  split
  · rename_i h
    have hz := topExp_none ts h
    have : sumBy Term.v ts = sumBy (fun _ => (0 : Int)) ts := sumBy_congr _ _ _ hz
    have h0 : ∀ l : List Term, sumBy (fun _ => (0 : Int)) l = 0 := by
      intro l
      induction l with
      | nil => rfl
      | cons _ _ ih => simp [ih]
    rw [← hsum, this, h0]
  · rename_i E hE
    have hEle : E ≤ 23 + L := by
      apply topExp_le (23 + L) ts _ E hE
      intro t ht hne
      obtain ⟨q, hv, hq, he⟩ := hts t ht
      exact exp_le L t q hv hq hne (he hne)
    have hW : withFloor u.floor E ≤ 23 + L := by
      unfold withFloor
      split
      · exact hEle
      · rename_i f hf
        have := hfl f hf
        simp only [Int.max_def]; split <;> omega
    have hcut : sumBy (fun t => cut u.cutMode t.v (withFloor u.floor E - u.F)) ts =
        sumBy Term.v ts := by
      apply sumBy_congr
      intro t ht
      obtain ⟨q, hv, _, _⟩ := hts t ht
      exact cut_id _ _ _ L (by omega) ⟨q, by rw [hv, Int.mul_comm]⟩
    rw [hcut, hsum, hround n hn]

/-! ## T1 -/

/-- A product term: integer value `p` (a digit product) with the exponent `e` the unit reads. -/
def prodTerm (L : Nat) (x : Int × Int) : Term := ⟨x.1 * 2 ^ L, x.2⟩

/-- Absolute sum of the integer products. -/
def absSum (xs : List (Int × Int)) : Int := sumBy (fun x => (x.1.natAbs : Int)) xs

/-- **T1, chain form.** Start the accumulator at an integer `a` with every partial sum in
`[2^23, 2^24)` (implied by `2^23 <= a - R` and `a + R < 2^24` with `R` the absolute sum of all
products). Then for any split of the products into groups (any group size, instruction shape),
any order, and any unit with `F >= 23`, the chain returns `a + sum of products` exactly. -/
theorem chain_exact (u : MatrixUnit) (round : Int → Int) (cexp : Int → Int) (L : Nat)
    (hF : 23 ≤ u.F) (hfl : ∀ f, u.floor = some f → f ≤ 23 + L)
    (hround : ∀ n : Int, n.natAbs < 2 ^ 24 → round (n * 2 ^ L) = n * 2 ^ L)
    (hc : ∀ c : Int, c ≠ 0 → cexp c ≤ (Nat.log2 c.natAbs : Int)) :
    ∀ (gs : List (List (Int × Int))),
      (∀ x ∈ gs.flatten, x.1 ≠ 0 → x.2 ≤ (Nat.log2 (x.1 * 2 ^ L).natAbs : Int)) →
      ∀ a : Int, 2 ^ 23 ≤ a - absSum gs.flatten → a + absSum gs.flatten < 2 ^ 24 →
      chain u round cexp (a * 2 ^ L) (gs.map (List.map (prodTerm L))) =
        (a + sumBy Prod.fst gs.flatten) * 2 ^ L := by
  intro gs
  induction gs with
  | nil => intro _ a _ _; simp [chain]
  | cons g gs ih =>
    intro hexp a hlo hhi
    have hsplit : absSum (g :: gs).flatten = absSum g + absSum gs.flatten := by
      simp only [absSum]; exact sumBy_flatten_cons _ g gs
    have hgs : -absSum g ≤ sumBy Prod.fst g ∧ sumBy Prod.fst g ≤ absSum g := sumBy_abs_le Prod.fst g
    have hR0 : 0 ≤ absSum g := sumBy_nonneg _ _ (fun _ _ => by omega)
    have hR1 : 0 ≤ absSum gs.flatten := sumBy_nonneg _ _ (fun _ _ => by omega)
    simp only [List.map_cons, chain]
    have hstep : group u round (⟨a * 2 ^ L, cexp (a * 2 ^ L)⟩ :: g.map (prodTerm L)) =
        (a + sumBy Prod.fst g) * 2 ^ L := by
      apply group_exact u round L hF hfl hround
      · intro t ht
        rcases List.mem_cons.mp ht with rfl | ht'
        · refine ⟨a, rfl, by omega, ?_⟩
          intro hne; exact hc _ hne
        · obtain ⟨x, hx, rfl⟩ := List.mem_map.mp ht'
          have hxa : (x.1.natAbs : Int) ≤ absSum g :=
            le_sumBy_of_mem _ g (fun _ _ => by omega) x hx
          refine ⟨x.1, rfl, by omega, ?_⟩
          intro hne
          exact hexp x (List.mem_flatten.mpr ⟨g, by simp, hx⟩) (by
            intro h0; apply hne; simp [prodTerm, h0])
      · simp only [sumBy_cons]
        have : sumBy Term.v (g.map (prodTerm L)) = sumBy Prod.fst g * 2 ^ L := by
          rw [← sumBy_mul_right]; simp [sumBy, prodTerm, List.map_map, Function.comp_def]
        rw [this, Int.add_mul]
      · omega
    rw [hstep]
    have hexp' : ∀ x ∈ gs.flatten, x.1 ≠ 0 → x.2 ≤ (Nat.log2 (x.1 * 2 ^ L).natAbs : Int) :=
      fun x hx => hexp x (by simp only [List.flatten_cons]; exact List.mem_append_right _ hx)
    rw [ih hexp' (a + sumBy Prod.fst g) (by omega)
      (by omega)]
    rw [sumBy_flatten_cons]
    congr 1
    omega

/-- **T1, any order.** If the products fed to the chain, in whatever grouping and order, are a
permutation of a list `ps`, the result is `C + sum ps`, exactly. -/
theorem chain_exact_perm (u : MatrixUnit) (round : Int → Int) (cexp : Int → Int) (L : Nat)
    (hF : 23 ≤ u.F) (hfl : ∀ f, u.floor = some f → f ≤ 23 + L)
    (hround : ∀ n : Int, n.natAbs < 2 ^ 24 → round (n * 2 ^ L) = n * 2 ^ L)
    (hc : ∀ c : Int, c ≠ 0 → cexp c ≤ (Nat.log2 c.natAbs : Int))
    (ps : List (Int × Int)) (gs : List (List (Int × Int))) (hperm : gs.flatten.Perm ps)
    (hexp : ∀ x ∈ ps, x.1 ≠ 0 → x.2 ≤ (Nat.log2 (x.1 * 2 ^ L).natAbs : Int))
    (C : Int) (hlo : 2 ^ 23 ≤ C - absSum ps) (hhi : C + absSum ps < 2 ^ 24) :
    chain u round cexp (C * 2 ^ L) (gs.map (List.map (prodTerm L))) =
      (C + sumBy Prod.fst ps) * 2 ^ L := by
  have h1 : absSum gs.flatten = absSum ps := sumBy_perm _ hperm
  have h2 : sumBy Prod.fst gs.flatten = sumBy Prod.fst ps := sumBy_perm _ hperm
  rw [chain_exact u round cexp L hF hfl hround hc gs
    (fun x hx => hexp x (hperm.subset hx)) C (by omega) (by omega), h2]

/-- **T1, the window.** Every accumulator value the chain passes through (after any prefix of
the groups) is `C` plus the partial sum so far, an integer in `[2^23, 2^24)`. -/
theorem chain_window (u : MatrixUnit) (round : Int → Int) (cexp : Int → Int) (L : Nat)
    (hF : 23 ≤ u.F) (hfl : ∀ f, u.floor = some f → f ≤ 23 + L)
    (hround : ∀ n : Int, n.natAbs < 2 ^ 24 → round (n * 2 ^ L) = n * 2 ^ L)
    (hc : ∀ c : Int, c ≠ 0 → cexp c ≤ (Nat.log2 c.natAbs : Int))
    (gs : List (List (Int × Int)))
    (hexp : ∀ x ∈ gs.flatten, x.1 ≠ 0 → x.2 ≤ (Nat.log2 (x.1 * 2 ^ L).natAbs : Int))
    (C : Int) (hlo : 2 ^ 23 ≤ C - absSum gs.flatten) (hhi : C + absSum gs.flatten < 2 ^ 24)
    (k : Nat) :
    chain u round cexp (C * 2 ^ L) ((gs.take k).map (List.map (prodTerm L))) =
        (C + sumBy Prod.fst (gs.take k).flatten) * 2 ^ L ∧
      2 ^ 23 ≤ C + sumBy Prod.fst (gs.take k).flatten ∧
      C + sumBy Prod.fst (gs.take k).flatten < 2 ^ 24 := by
  have hsplit : absSum gs.flatten = absSum (gs.take k).flatten + absSum (gs.drop k).flatten := by
    conv => lhs; rw [← List.take_append_drop k gs]
    simp only [absSum, List.flatten_append, sumBy_append]
  have hd : 0 ≤ absSum (gs.drop k).flatten := sumBy_nonneg _ _ (fun _ _ => by omega)
  have ht : 0 ≤ absSum (gs.take k).flatten := sumBy_nonneg _ _ (fun _ _ => by omega)
  have hab : -absSum (gs.take k).flatten ≤ sumBy Prod.fst (gs.take k).flatten ∧ sumBy Prod.fst (gs.take k).flatten ≤ absSum (gs.take k).flatten := sumBy_abs_le Prod.fst (gs.take k).flatten
  refine ⟨?_, by omega, by omega⟩
  apply chain_exact u round cexp L hF hfl hround hc
  · intro x hx; apply hexp x
    have : (gs.take k).flatten ⊆ gs.flatten := by
      intro y hy
      obtain ⟨l, hl, hyl⟩ := List.mem_flatten.mp hy
      exact List.mem_flatten.mpr ⟨l, List.mem_of_mem_take hl, hyl⟩
    exact this hx
  · omega
  · omega

end Truestar.TC
